"""
Gamma BCANN: model, guide and SVI training loop.

Gamma (claim severity) counterpart of poisson_bcann_vi.py.

    eta = log(y_scale) + x'beta + b0 + f(x; w),   mu = exp(eta)
    y_i ~ Gamma(shape = alpha * w_i, rate = alpha * w_i / mu_i)

so E[y_i] = mu_i and Var[y_i] = mu_i^2 / (alpha * w_i), where
    y_i   : average claim size of policy i
    w_i   : number of claims behind that average (the GLM weight)
    alpha : Gamma shape (1 / dispersion), Bayesian with a LogNormal prior; q(alpha) starts
            at the GLM Pearson estimate and is learned jointly with the network weights
            after a warm-up with alpha held fixed (see gamma_bcann_svi_training_loop)
    beta, b0, w : Bayesian (BayesianLinearPredictor / BayesianMLP priors)

Data loaders yield (X, y, w) where Poisson loaders yield (X, y, log_exposure).

Usage (python gamma_bcann_vi.py [--data fremtpl|synthetic]):
    model   = GammaBCANN(48, nn_params, y_scale=gamma_weighted_mean(y_train))
    glm_mle = gamma_fit_glm_mle(X_train, y_train, w_train, y_scale=model.y_scale)
    guide   = gamma_build_bcann_guide(model, X_train[:256], y_train[:256], w_train[:256], glm_mle=glm_mle)
    history = gamma_bcann_svi_training_loop(model, guide, train_loader, val_loader, n_epochs=300)
"""
import math
import os
import re

import torch
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import (
    AutoGuideList,
    AutoMultivariateNormal,
    AutoLowRankMultivariateNormal,
    AutoNormal
)
from pyro.infer.autoguide.initialization import init_to_value, init_to_sample
from pyro.nn import PyroModule
from pyro.optim import ClippedAdam
from codebase_0 import BayesianCANN, CANN, set_seed, LinearPredictor
from poisson_bcann_vi import (
    _expose, _snapshot_params, _restore_params, _resolve_device, kl_weight, _kl_scaled
)

SHAPE_SITE = "gamma_shape"

# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #

def gamma_deviance(y_true, y_pred, weights=None, eps=1e-8):
    """
    Weighted mean Gamma deviance  sum_i w_i d(y_i, mu_i) / sum_i w_i,
        d(y, mu) = 2 [ (y - mu) / mu - log(y / mu) ]

    y_pred: eta = log(mu), not mu (same convention as poisson_deviance)
    weights: number of claims per observation (None -> all 1)
    """
    eta = y_pred.clamp(min=-30.0, max=30.0)
    log_ratio = torch.log(y_true.clamp(min=eps)) - eta          # log(y / mu)
    d = 2 * (torch.exp(log_ratio) - 1 - log_ratio)
    if weights is None:
        return d.mean()
    return (weights * d).sum() / weights.sum()


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class GammaBCANN(PyroModule):
    """
    eta = log(y_scale) + x'beta + b0 + f(x; w),   mu = exp(eta)
    y ~ Gamma(alpha * w, alpha * w / mu),   alpha ~ LogNormal(shape_prior_loc, shape_prior_scale)

    y_scale is a fixed offset (use the weighted mean claim size of the training set) so the
    N(0, 1) prior on the intercept is centred near the right order of magnitude; without it the
    intercept would have to reach log(~2000) ~ 7.6 prior standard deviations away from zero.

    kl_weight scales the priors (beta, b0, w and alpha) and nothing else, as in PoissonBCANN;
    the likelihood keeps weight 1. Pair it with _kl_scaled(guide) for minibatch KL weighting.
    """

    def __init__(self, in_features, neural_net_params, y_scale=1.0,
                 shape_prior_loc=0.0, shape_prior_scale=1.0):
        super().__init__()
        self.cann = BayesianCANN(in_features, neural_net_params)
        self.y_scale = float(y_scale)
        self.shape_prior_loc = shape_prior_loc
        self.shape_prior_scale = shape_prior_scale

    def forward(self, X, y=None, weights=None, kl_weight=1.0):
        offset = math.log(self.y_scale)
        # PyroSample sites are sampled on first access inside these calls,
        # so this scales log p(beta, b0, w, alpha) by kl_weight and nothing else.
        with poutine.scale(scale=kl_weight):
            eta_glm = offset + self.cann.linear_pred(X)
            f_nn = self.cann.neural_net(X)
            shape = pyro.sample(SHAPE_SITE, dist.LogNormal(
                torch.tensor(self.shape_prior_loc, device=X.device),
                torch.tensor(self.shape_prior_scale, device=X.device)))
        eta = eta_glm + f_nn
        mu = torch.exp(eta.clamp(min=-30.0, max=30.0))           # expected claim size

        concentration = shape if weights is None else shape * weights

        # Kept as sites so predictive draws can be split into GLM / network parts.
        pyro.deterministic("eta_glm", eta_glm)
        pyro.deterministic("mu", mu)

        with pyro.plate("data", X.shape[0]):          # no scale: likelihood weight stays 1
            pyro.sample("obs", dist.Gamma(concentration, concentration / mu), obs=y)
        return mu


# --------------------------------------------------------------------------- #
# GLM warm start (MLE) and guide
# --------------------------------------------------------------------------- #

def gamma_weighted_mean(y, weights=None):
    return (y.mean() if weights is None else (weights * y).sum() / weights.sum()).item()


def gamma_fit_glm_mle(X, y, weights=None, y_scale=1.0, random_state=None, max_iter=200):
    """
    Gamma GLM (log link) on the same offset as GammaBCANN.
    Returns (weight, bias, shape); shape is the Pearson estimate 1 / phi_hat.

    beta does not depend on alpha (mean and shape are Fisher-orthogonal), so the GLM fit is
    exact without it. The Pearson shape comes from GLM-only residuals, before the network has
    absorbed any non-linear signal, so it is biased low: use it as an initial value only.
    """
    if random_state is not None:
        torch.manual_seed(random_state)
    glm = LinearPredictor(X.shape[1])
    # glm lands on torch's default device (e.g. cuda after torch.set_default_device), so
    # fit on whatever device that is rather than the data's
    p = next(glm.parameters())
    X, y = X.to(p.device, p.dtype), y.to(p.device, p.dtype)
    w = torch.ones_like(y) if weights is None else weights.to(p.device, p.dtype)
    offset = math.log(y_scale)

    # Start at the intercept-only MLE: log(weighted mean claim size)
    with torch.no_grad():
        glm.linear.weight.zero_()
        glm.linear.bias.fill_(math.log(gamma_weighted_mean(y, w)) - offset)

    opt = torch.optim.LBFGS(glm.parameters(), lr=1.0, max_iter=max_iter,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = gamma_deviance(y_true=y, y_pred=glm(X) + offset, weights=w)
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        mu = torch.exp(glm(X) + offset)
        phi = (w * ((y - mu) / mu) ** 2).sum() / (y.numel() - X.shape[1] - 1)
    return (glm.linear.weight.detach().clone(), glm.linear.bias.detach().clone(),
            (1.0 / phi).reshape(()).detach().clone())


def _gamma_latent_sites(model, X, y, weights=None):
    """
    Returns latent named sites of the model
    """
    with torch.no_grad():
        tr = poutine.trace(model).get_trace(X, y, weights=weights)
    return {n: s["value"] for n, s in tr.nodes.items()
            if s["type"] == "sample" and not s["is_observed"]
            and ("linear_pred" in n or "neural_net" in n or n == SHAPE_SITE)}


def gamma_build_bcann_guide(model, X, y, weights=None, glm_mle=None, rank=10, init_scale=0.05):
    """
    q(beta, b0) = full-covariance Gaussian          (AutoMultivariateNormal)
    q(alpha)    = log-normal                        (AutoNormal on log alpha)
    q(w)        = low-rank + diagonal Gaussian      (AutoLowRankMultivariateNormal)
    q = q(beta, b0) * q(alpha) * q(w)

    Variational means are initialised so that the model starts as the Gamma GLM:
      - GLM block (and alpha) at the MLE (if glm_mle given), otherwise a prior draw
      - final network layer at exactly zero, so f(x; w) starts at 0 (CANN zero-init)
      - remaining hidden-layer means at a prior draw (non-zero, to break symmetry)
    """
    sites = _gamma_latent_sites(model, X, y, weights)
    values = {}

    if glm_mle is not None:
        w_mle, b_mle = glm_mle[0], glm_mle[1]
        for n in sites:
            if "linear_pred" in n:
                values[n] = w_mle if n.endswith("weight") else b_mle
        if len(glm_mle) > 2:
            values[SHAPE_SITE] = glm_mle[2]

    layer_idx = [int(m.group(1)) for n in sites
                 if (m := re.search(r"neural_net\.net\.(\d+)\.", n))]
    last = max(layer_idx)
    for n, v in sites.items():
        if f"neural_net.net.{last}." in n:
            values[n] = torch.zeros_like(v)

    init_fn = init_to_value(values=values, fallback=init_to_sample)

    glm_guide = AutoMultivariateNormal(
        poutine.block(model, expose_fn=_expose("linear_pred")),
        init_loc_fn=init_fn, init_scale=init_scale)
    shape_guide = AutoNormal(
        poutine.block(model, expose_fn=_expose(SHAPE_SITE)),
        init_loc_fn=init_fn, init_scale=init_scale)
    nn_guide = AutoLowRankMultivariateNormal(
        poutine.block(model, expose_fn=_expose("neural_net")),
        init_loc_fn=init_fn, init_scale=init_scale, rank=rank)

    guide = AutoGuideList(model)
    guide.append(glm_guide)
    guide.append(shape_guide)
    guide.append(nn_guide)
    guide(X, y, weights=weights)   # run once so the prototype / parameters are created
    return guide


# --------------------------------------------------------------------------- #
# Prediction and evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def gamma_posterior_mu(model, guide, X, num_samples=50):
    """Posterior draws of mu = E[Y | x, theta], shape [num_samples, n]."""
    pred = Predictive(model, guide=guide, num_samples=num_samples, return_sites=("mu",))
    return pred(X)["mu"]


@torch.no_grad()
def gamma_posterior_shape(model, guide, X, num_samples=500):
    """Posterior draws of the Gamma shape alpha, shape [num_samples]."""
    pred = Predictive(model, guide=guide, num_samples=num_samples, return_sites=(SHAPE_SITE,))
    return pred(X[:1])[SHAPE_SITE].reshape(-1)


@torch.no_grad()
def gamma_evaluate_bcann(model, guide, loader, num_samples=50, device=None):
    """Weighted mean Gamma deviance of the posterior-mean claim size E_q[mu]."""
    device = _resolve_device(device)
    mus, ys, ws = [], [], []
    for Xb, yb, wb in loader:
        Xb, yb, wb = Xb.to(device), yb.to(device), wb.to(device)
        mu_bar = gamma_posterior_mu(model, guide, Xb, num_samples).mean(0).reshape(-1)  # (batch,)
        assert mu_bar.shape == yb.shape, (mu_bar.shape, yb.shape)
        mus.append(mu_bar)
        ys.append(yb)
        ws.append(wb)
    mu, y, w = torch.cat(mus), torch.cat(ys), torch.cat(ws)
    return gamma_deviance(y_true=y, y_pred=torch.log(mu), weights=w).item()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def _gamma_check_data(loader):
    """Fail early, with a readable message, if y or the claim-count weights are not positive."""
    _, y, w = loader.dataset.tensors
    if (y <= 0).any():
        raise ValueError(f"Gamma needs claim sizes y > 0; found min(y) = {y.min().item():.4g}")
    if (w <= 0).any():
        raise ValueError(
            f"Gamma weights must be claim counts (>= 1); found min(w) = {w.min().item():.4g}. "
            "Is the third tensor exposure / log-exposure? For one row per claim use w = 1.")


@torch.no_grad()
def _gamma_shape_median(guide, X, y, weights=None):
    return guide.median(X, y, weights=weights)[SHAPE_SITE].detach().clone()


def gamma_bcann_svi_training_loop(model, guide, train_loader, val_loader, n_epochs,
                            lr=1e-2, clip_norm=10.0, num_particles=1,
                            val_every=1, val_samples=50, patience=10, min_delta=1e-6,
                            verbose_every=25, random_state=None, device=None,
                            kl_scheme="blundell", shape_warmup_epochs=20, shape_lr_mult=0.1, shape_init=None,
                            track_train=True):
    """
    Bayes-by-backprop style SVI. Early stopping on validation Gamma deviance of the
    posterior-mean prediction, checked every `val_every` epochs; `patience` is counted
    in validation checks. Best variational parameters are restored at the end.

    The Gamma shape alpha is learned jointly with the network weights, in two stages:
      1. epochs 1..shape_warmup_epochs: alpha is held at alpha_0 (the model is conditioned
         on it and q(alpha) is not updated) while q(beta, b0) and q(w) train. alpha_0 is
         `shape_init`, or the current median of q(alpha), i.e. the GLM Pearson estimate
         when the guide was built with glm_mle.
      2. afterwards q(alpha) is trained jointly in the ELBO at lr * shape_lr_mult.
    alpha enters the ELBO as a likelihood temperature, so learning it while the network is
    still poor pulls it low (the GLM-only residuals over-state dispersion), which weakens the
    likelihood against the KL term and can collapse q(w) onto the prior; the warm-up and the
    smaller learning rate avoid that. Early stopping and best-state tracking only start after
    the warm-up, so the restored state always has a learned alpha.

    Minibatch i of M has its prior and guide terms (the KL) scaled by kl_weight(i, M, kl_scheme)
    ("blundell" or "uniform"), as in the Poisson loop; the weights sum to 1 over an epoch, so
    one epoch adds up to one full ELBO.

    `device=None` uses torch's current default device. Batches are moved to it; the model and
    guide must already live there (build the guide with data on that device).

    track_train : also compute the same deviance on train_loader at every validation check
        (history["train_deviance"], aligned with history["val_epochs"]); costs one extra
        prediction pass over the training data per check.
    """
    set_seed(random_state)
    _gamma_check_data(train_loader)
    _gamma_check_data(val_loader)
    device = _resolve_device(device)
    n_train = len(train_loader.dataset)

    optim = ClippedAdam(lambda name: {
        "lr": lr * (shape_lr_mult if SHAPE_SITE in name else 1.0), "clip_norm": clip_norm})
    loss = Trace_ELBO(num_particles=num_particles)
    svi = SVI(model, _kl_scaled(guide), optim, loss=loss)

    if shape_warmup_epochs > 0:
        if shape_init is None:
            Xb, yb, wb = next(iter(train_loader))
            shape_init = _gamma_shape_median(guide, Xb.to(device), yb.to(device), wb.to(device))
        shape_init = torch.as_tensor(shape_init, dtype=torch.get_default_dtype(),
                                     device=device).reshape(())
        svi_warmup = SVI(poutine.condition(model, data={SHAPE_SITE: shape_init}),
                         # also hides AutoNormal's auxiliary "<site>_unconstrained" sample,
                         # whose entropy term would otherwise still move q(alpha)'s scale
                         _kl_scaled(poutine.block(guide, hide_fn=lambda msg: msg["type"] == "sample"
                                                  and SHAPE_SITE in msg["name"])),
                         optim, loss=loss)
        print(f"Shape warm-up: alpha held at {shape_init.item():.3f} "
              f"for {shape_warmup_epochs} epochs")

    history = {"neg_elbo_per_obs": [], "val_epochs": [], "val_deviance": [], "train_deviance": [],
               "val_shape": [], "best_val_deviance": float("inf"), "best_epoch": 0}
    best_state, bad_checks = None, 0

    for epoch in range(1, n_epochs + 1):
        warmup = epoch <= shape_warmup_epochs
        step = (svi_warmup if warmup else svi).step
        M = len(train_loader)
        batch_losses = [step(Xb.to(device), yb.to(device), weights=wb.to(device),
                             kl_weight=kl_weight(i, M, kl_scheme))
                        for i, (Xb, yb, wb) in enumerate(train_loader, start=1)]
        history["neg_elbo_per_obs"].append(sum(batch_losses) / n_train)      # SUM, not mean

        if epoch % val_every == 0 or epoch == 1:
            val_dev = gamma_evaluate_bcann(model, guide, val_loader, val_samples, device)
            Xv, yv, wv = next(iter(val_loader))
            shape = _gamma_shape_median(guide, Xv[:1].to(device), yv[:1].to(device),
                                        wv[:1].to(device)).item()
            history["val_epochs"].append(epoch)
            history["val_deviance"].append(val_dev)
            history["val_shape"].append(shape)
            if track_train:
                history["train_deviance"].append(
                    gamma_evaluate_bcann(model, guide, train_loader, val_samples, device))

            if warmup:
                pass
            elif val_dev < history["best_val_deviance"] - min_delta:
                history["best_val_deviance"], history["best_epoch"] = val_dev, epoch
                if track_train:
                    history["best_train_deviance"] = history["train_deviance"][-1]
                best_state, bad_checks = _snapshot_params(), 0
            else:
                bad_checks += 1

            if epoch == 1 or epoch % verbose_every == 0:
                print(f"Epoch {epoch} | -ELBO/n: {history['neg_elbo_per_obs'][-1]:.4f} "
                      + (f"| train deviance: {history['train_deviance'][-1]:.4f} " if track_train else "")
                      + f"| val deviance: {val_dev:.4f} | alpha (median): {shape:.3f}"
                      + (" [warm-up]" if warmup else ""))

            if bad_checks >= patience:
                print(f"Early stopping at epoch {epoch}. Best val deviance "
                      f"{history['best_val_deviance']:.4f} at epoch {history['best_epoch']}")
                break

    if best_state is not None:
        _restore_params(best_state)
    return history


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

class GammaCANN(CANN):
    """
    CANN with a fixed log(y_scale) offset, as in GammaBCANN, so the GLM intercept and the
    network work on the scale of claim size relative to y_scale. forward returns eta.
    """

    def __init__(self, in_features, neural_net_params, y_scale=1.0, random_state=None):
        super().__init__(in_features, neural_net_params, random_state=random_state)
        self.y_scale = float(y_scale)

    def forward(self, X):
        return math.log(self.y_scale) + super().forward(X)


# --------------------------------------------------------------------------- #
# GammaCANN class and training loop
# --------------------------------------------------------------------------- #

def gamma_mlp_train_loop(model, optimizer, n_epochs, train_loader, val_loader, loss_fn=gamma_deviance,
                         device="cpu", patience=20, min_delta=1e-9, verbose_every=100):
    """
    codebase_0.mlp_train_loop with claim-count weights: loaders yield (X, y, w) and
    loss_fn(y_true, y_pred, weights) is a weighted mean, so epoch losses are weighted by
    sum(w) per batch and equal the full-data weighted deviance.
    """
    model.to(device)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state, best_epoch = None, 0
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        ### Training
        model.train()
        train_loss, w_train = 0.0, 0.0
        for X_batch, y_batch, w_batch in train_loader:
            X_batch, y_batch, w_batch = X_batch.to(device), y_batch.to(device), w_batch.to(device)

            y_pred = model(X_batch)
            loss = loss_fn(y_true=y_batch, y_pred=y_pred, weights=w_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * w_batch.sum().item()
            w_train += w_batch.sum().item()
        train_loss /= w_train

        ### Validation
        model.eval()
        val_loss, w_val = 0.0, 0.0
        with torch.inference_mode():
            for X_batch, y_batch, w_batch in val_loader:
                X_batch, y_batch, w_batch = X_batch.to(device), y_batch.to(device), w_batch.to(device)

                val_pred = model(X_batch)
                batch_loss = loss_fn(y_true=y_batch, y_pred=val_pred, weights=w_batch)

                val_loss += batch_loss.item() * w_batch.sum().item()
                w_val += w_batch.sum().item()
        val_loss /= w_val

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        ### Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % verbose_every == 0 or epoch == 0:
            print(f"Epoch {epoch + 1} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}. "
                  f"Best val loss {best_val_loss:.4f} at epoch {best_epoch}")
            break

    ### Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


@torch.no_grad()
def gamma_cann_pearson_shape(model, X, y, weights=None):
    """Pearson estimate of the Gamma shape 1 / phi_hat from the fitted CANN."""
    device = next(model.parameters()).device
    X, y = X.to(device), y.to(device)
    w = torch.ones_like(y) if weights is None else weights.to(device)
    mu = torch.exp(model(X).clamp(min=-30.0, max=30.0))
    n_params = sum(p.numel() for p in model.parameters())
    phi = (w * ((y - mu) / mu) ** 2).sum() / max(y.numel() - n_params, 1)
    return (1.0 / phi).item()


def gamma_cann_training_loop(cann_model, nn_optimiser, nn_optim_params, nn_epochs,
                             train_loader, val_loader, early_stop_params=None,
                             device="cpu", random_state=None):
    """
    Gamma version of codebase_0.cann_training_loop:
      1. fit the GLM alone (gamma_fit_glm_mle, LBFGS on the weighted Gamma deviance)
      2. freeze the GLM
      3. train the network on the weighted Gamma deviance of GLM + NN
    The returned history also holds the GLM Pearson shape and the CANN Pearson shape
    (on the training data).
    """
    set_seed(random_state)
    _gamma_check_data(train_loader)
    _gamma_check_data(val_loader)
    X_all, y_all, w_all = train_loader.dataset.tensors

    # Step 1: fit GLM alone, then copy into the CANN's GLM
    glm_weight, glm_bias, glm_shape = gamma_fit_glm_mle(
        X_all, y_all, w_all, y_scale=cann_model.y_scale, random_state=random_state)
    with torch.no_grad():
        cann_model.glm.linear.weight.copy_(glm_weight)
        cann_model.glm.linear.bias.copy_(glm_bias)

    # Step 2: freeze GLM
    for p in cann_model.glm.parameters():
        p.requires_grad = False

    # Step 3: train network on residual structure (GLM + NN output)
    cann_model.to(device)
    optimiser = nn_optimiser(cann_model.neural_net.parameters(), **nn_optim_params)
    history = gamma_mlp_train_loop(cann_model, optimiser, nn_epochs, train_loader, val_loader,
                                   device=device, **(early_stop_params or {}))

    history["glm_shape"] = glm_shape.item()
    history["shape"] = gamma_cann_pearson_shape(cann_model, X_all, y_all, w_all)
    return history