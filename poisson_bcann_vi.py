"""
Poisson BCANN: model, guide and SVI training loop.

Assumes your existing definitions are importable from the same namespace
(set_seed, LinearPredictor, BayesianCANN, poisson_deviance).
e.g. put this below them in the notebook, or `from your_module import *`.
"""
import copy
import re

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import (
    AutoGuideList,
    AutoMultivariateNormal,
    AutoLowRankMultivariateNormal,
    AutoNormal,
)
from pyro.infer.autoguide.initialization import init_to_value, init_to_sample
from pyro.nn import PyroModule
from pyro.optim import ClippedAdam
from codebase_0 import (
    BayesianCANN, poisson_deviance, set_seed, LinearPredictor, MLP
)

def _resolve_device(device=None):
    """Explicit device if given, else whatever torch's current default device is."""
    return torch.device(device) if device is not None else torch.empty(0).device


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class PoissonBCANN(PyroModule):
    """
    eta = x'beta + b0 + f(x; w),   mu = exp(eta),   y ~ Poisson(mu)

    beta, b0 ~ prior (BayesianLinearPredictor); w ~ prior (BayesianMLP).
    No exposure offset (matches poisson_deviance); add log-exposure to eta
    here and in fit_glm_mle if you need it.
    """

    def __init__(self, in_features, neural_net_params):
        super().__init__()
        self.cann = BayesianCANN(in_features, neural_net_params)

    def forward(self, X, y=None, log_exposure=None, kl_weight=1.0):
    # PyroSample sites are sampled on first access inside these calls,
    # so this scales log p(beta, b0, w) by kl_weight and nothing else.
        with poutine.scale(scale=kl_weight):
            eta_glm = self.cann.linear_pred(X)
            f_nn = self.cann.neural_net(X)
        eta = eta_glm + f_nn
        mu = torch.exp(eta.clamp(max=20.0))
        rate = mu if log_exposure is None else torch.exp((eta + log_exposure).clamp(max=20.0))

        pyro.deterministic("eta_glm", eta_glm)
        pyro.deterministic("mu", mu)
        pyro.deterministic("rate", rate)

        with pyro.plate("data", X.shape[0]):          # no scale: likelihood weight stays 1
            pyro.sample("obs", dist.Poisson(rate), obs=y)
        return mu


# --------------------------------------------------------------------------- #
# GLM warm start (MLE) and guide
# --------------------------------------------------------------------------- #

def kl_weight(i, M, scheme="blundell"):
    """KL weight for minibatch i (1-indexed) of M; sums to 1 over an epoch."""
    if scheme == "blundell":
        pi = 2.0 ** (-i) / (1.0 - 2.0 ** (-M))    # = 2^(M-i) / (2^M - 1)
    elif scheme == "uniform":
        pi = 1.0 / M
    else:
        raise ValueError(scheme)
    return max(pi, 1e-30)   # poutine.scale requires scale > 0; 2**-i underflows to 0 for i > 1074

def _kl_scaled(guide):
    def scaled_guide(*args, kl_weight=1.0, **kwargs):
        with poutine.scale(scale=kl_weight):      # scales log q(z) and the Delta Jacobian terms
            return guide(*args, **kwargs)
    return scaled_guide

def poisson_fit_glm_mle(X, y, log_exposure=None, random_state=None, max_iter=200):
    if random_state is not None:
        torch.manual_seed(random_state)
    glm = LinearPredictor(X.shape[1])            # however you construct it
    dtype = next(glm.parameters()).dtype
    X, y = X.to(dtype), y.to(dtype)
    off = torch.zeros_like(y) if log_exposure is None else log_exposure.to(dtype)

    # Start at the intercept-only MLE: log(total claims / total exposure)
    with torch.no_grad():
        glm.linear.weight.zero_()
        glm.linear.bias.fill_(torch.log(y.sum() / off.exp().sum()))

    opt = torch.optim.LBFGS(glm.parameters(), lr=1.0, max_iter=max_iter,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = poisson_deviance(y_true=y, y_pred=glm(X) + off)   # off = log_exposure
        loss.backward()
        return loss

    opt.step(closure)
    return glm.linear.weight.detach().clone(), glm.linear.bias.detach().clone()

# A site in Pyro is a random variable is sampled by Pyro

def _latent_sites(model, X, y):
    """
    Returns latent named sites of the model
    """
    with torch.no_grad():
        tr = poutine.trace(model).get_trace(X, y)
    return {n: s["value"] for n, s in tr.nodes.items()
            if s["type"] == "sample" and not s["is_observed"]
            and ("linear_pred" in n or "neural_net" in n)}   # excludes the plate pseudo-site

def _expose(substr):
    """
    Returns a filter function that decides which side a sub-guide is allowed to see.
    Use in adjunct with poutine.block to create separate "guides" for latent sites
    """
    return lambda msg: (msg["type"] == "sample" and not msg["is_observed"]
                        and substr in msg["name"])


def build_bcann_guide(model, X, y, glm_mle=None, rank=10, init_scale=0.05):
    """
    q(beta, b0) = full-covariance Gaussian
    q(w)        = low-rank + diagonal Gaussian
    q = q(beta, b0) * q(w)   (the independence assumption the variance decomposition relies on)

    Variational means are initialised so that the model starts as the GLM:
      - GLM block at the MLE (if glm_mle given), otherwise a prior draw
      - final network layer at exactly zero, so f(x; w) starts at 0 (CANN zero-init)
      - remaining hidden-layer means at a prior draw (non-zero, to break symmetry)
    """
    sites = _latent_sites(model, X, y)
    values = {}

    # if provided, the GLM coefficients are 
    if glm_mle is not None:
        w_mle, b_mle = glm_mle
        for n in sites:
            if "linear_pred" in n:
                values[n] = w_mle if n.endswith("weight") else b_mle

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
    nn_guide = AutoLowRankMultivariateNormal(
        poutine.block(model, expose_fn=_expose("neural_net")),
        init_loc_fn=init_fn, init_scale=init_scale, rank=rank)

    guide = AutoGuideList(model)
    guide.append(glm_guide)
    guide.append(nn_guide)
    guide(X, y)   # run once so the prototype / parameters are created
    return guide


def _is_shrinkage(name):
    """Horseshoe scale sites from HorseshoeMLP: nn_tau, nn_c2, nn_lam_w_{i}."""
    return name.startswith("nn_")


def build_guide_horseshoe_bcann(model, X, y, glm_mle=None, rank=10, init_scale=0.05,
                                tau_init=None):
    """
    Guide for HorseshoePoissonBCANN.

    q(beta, b0)         = full-covariance Gaussian
    q(w)                = low-rank + diagonal Gaussian
    q(tau, c2, lambda)  = independent log-normals (mean-field)
    q = q(beta, b0) * q(w) * q(tau, c2, lambda)

    The shrinkage block is an AutoNormal on positive sites: it is Normal(loc, scale)
    in unconstrained space pushed through exp, i.e. exactly LogNormal(loc, scale).

    Initialisation is the same CANN start as build_bcann_guide (GLM at the MLE,
    final network layer at zero), with the shrinkage scales at their prior medians:
    tau = tau_0 (or `tau_init`), c2 = slab_scale^2, lambda = 1.
    """
    with torch.no_grad():
        tr = poutine.trace(model).get_trace(X, y)
    sites = {n: s["value"] for n, s in tr.nodes.items()
             if s["type"] == "sample" and not s["is_observed"]
             and ("linear_pred" in n or "neural_net" in n or _is_shrinkage(n))}
    values = {}

    if glm_mle is not None:
        w_mle, b_mle = glm_mle
        for n in sites:
            if "linear_pred" in n:
                values[n] = w_mle if n.endswith("weight") else b_mle

    layer_idx = [int(m.group(1)) for n in sites
                 if (m := re.search(r"neural_net\.net\.(\d+)\.", n))]
    last = max(layer_idx)
    for n, v in sites.items():
        if f"neural_net.net.{last}." in n:
            values[n] = torch.zeros_like(v)

    hs = model.cann.neural_net
    for n, v in sites.items():
        if not _is_shrinkage(n):
            continue
        if n == "nn_tau":
            values[n] = torch.full_like(v, hs.tau_0 if tau_init is None else tau_init)
        elif n == "nn_c2":
            values[n] = torch.full_like(v, hs.slab_scale ** 2)
        else:                                   # nn_lam_w_{i}
            values[n] = torch.ones_like(v)

    init_fn = init_to_value(values=values, fallback=init_to_sample)

    glm_guide = AutoMultivariateNormal(
        poutine.block(model, expose_fn=_expose("linear_pred")),
        init_loc_fn=init_fn, init_scale=init_scale)
    nn_guide = AutoLowRankMultivariateNormal(
        poutine.block(model, expose_fn=_expose("neural_net")),
        init_loc_fn=init_fn, init_scale=init_scale, rank=rank)
    # startswith, not _expose's substring test, so no site can fall into two blocks
    shrink_guide = AutoNormal(
        poutine.block(model, expose_fn=lambda msg: msg["type"] == "sample"
                      and not msg["is_observed"] and _is_shrinkage(msg["name"])),
        init_loc_fn=init_fn, init_scale=init_scale)

    guide = AutoGuideList(model)
    guide.append(glm_guide)
    guide.append(nn_guide)
    guide.append(shrink_guide)
    guide(X, y)   # run once so the prototype / parameters are created
    return guide


# --------------------------------------------------------------------------- #
# Prediction and evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def posterior_mu(model, guide, X, num_samples=50):
    """Posterior draws of mu = E[Y | x, theta], shape [num_samples, n]."""
    pred = Predictive(model, guide=guide, num_samples=num_samples, return_sites=("mu",))
    return pred(X)["mu"]


@torch.no_grad()
def evaluate_bcann(model, guide, loader, num_samples=50, device=None):
    """Mean Poisson deviance of the posterior-mean expected count E_q[mu] * exposure."""
    device = _resolve_device(device)
    rates, ys = [], []
    for Xb, yb, lb in loader:
        Xb, yb, lb = Xb.to(device), yb.to(device), lb.to(device)
        mu_bar = posterior_mu(model, guide, Xb, num_samples).mean(0).reshape(-1)  # (batch,)
        assert mu_bar.shape == yb.shape, (mu_bar.shape, yb.shape)
        rates.append(mu_bar * torch.exp(lb))
        ys.append(yb)
    rate, y = torch.cat(rates), torch.cat(ys)
    return poisson_deviance(y_true=y, y_pred=torch.log(rate)).item()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def _snapshot_params():
    """
    Returns a snapshot of the variational parameters, for later recovery
    """
    st = pyro.get_param_store().get_state()
    return {"params": {k: v.detach().clone() for k, v in st["params"].items()},
            "constraints": copy.copy(st["constraints"])}


def _restore_params(state):
    """
    Copy a _snapshot_params() snapshot back into the live variational parameters, in place.
    (ParamStore.set_state swaps the store's entries but the guide modules keep using their own
    tensors, so it would silently restore nothing.) The guide must already be built.
    """
    live = dict(pyro.get_param_store().named_parameters())
    missing = set(state["params"]) - set(live)
    if missing:
        raise KeyError(f"Guide parameters not found (is the guide built?): {sorted(missing)}")
    with torch.no_grad():
        for k, v in state["params"].items():
            live[k].copy_(v)


class _EarlyStopper:
    """Tracks the best validation score, snapshots params on improvement, signals when to stop."""

    def __init__(self, patience, min_delta):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.best_epoch = float("inf"), 0
        self.bad_checks, self.state = 0, None

    def update(self, val_dev, epoch):
        """Returns True if training should stop."""
        if val_dev < self.best - self.min_delta:
            self.best, self.best_epoch = val_dev, epoch
            self.state, self.bad_checks = _snapshot_params(), 0
        else:
            self.bad_checks += 1
        return self.bad_checks >= self.patience

    def restore(self):
        if self.state is not None:
            _restore_params(self.state)


def _train_one_epoch(svi, loader, n_train, device, scheme):
    """Returns -ELBO per observation."""
    M = len(loader)
    total = 0.0
    for i, (Xb, yb, lb) in enumerate(loader, start=1):
        Xb, yb, lb = Xb.to(device), yb.to(device), lb.to(device)
        total += svi.step(Xb, yb, log_exposure=lb, kl_weight=kl_weight(i, M, scheme))
    return total / n_train      # SUM, not mean


def poisson_bcann_svi_training_loop(model, guide, train_loader, val_loader, n_epochs,
                            lr=1e-2, clip_norm=10.0, num_particles=1,
                            val_every=1, val_samples=50, patience=10, min_delta=1e-6,
                            verbose_every=25, random_state=None, device=None,
                            kl_scheme="blundell", track_train=True):

    """
    Bayes-by-backprop style SVI. Early stopping on validation Poisson deviance of the
    posterior-mean prediction, checked every `val_every` epochs; `patience` is counted
    in validation checks. Best variational parameters are restored at the end.
    `device=None` uses torch's current default device.
    track_train : also compute the same deviance on train_loader at every validation check
        (history["train_deviance"], aligned with history["val_epochs"]); costs one extra
        prediction pass over the training data per check.
    """
    set_seed(random_state)
    device = _resolve_device(device)
    n_train = len(train_loader.dataset)

    svi = SVI(model, _kl_scaled(guide), ClippedAdam({"lr": lr, "clip_norm": clip_norm}),
              loss=Trace_ELBO(num_particles=num_particles))
    
    stopper = _EarlyStopper(patience, min_delta)

    history = {"neg_elbo_per_obs": [], "val_epochs": [], "val_deviance": [], "train_deviance": []}

    for epoch in range(1, n_epochs + 1):
        history["neg_elbo_per_obs"].append(_train_one_epoch(svi, train_loader, n_train, device, kl_scheme))

        if epoch != 1 and epoch % val_every != 0:
            continue

        val_dev = evaluate_bcann(model, guide, val_loader, val_samples, device)
        history["val_epochs"].append(epoch)
        history["val_deviance"].append(val_dev)
        if track_train:
            history["train_deviance"].append(evaluate_bcann(model, guide, train_loader, val_samples, device))
        should_stop = stopper.update(val_dev, epoch)

        if epoch == 1 or epoch % verbose_every == 0:
            print(f"Epoch {epoch} | -ELBO/n: {history['neg_elbo_per_obs'][-1]:.4f} "
                  + (f"| train deviance: {history['train_deviance'][-1]:.4f} " if track_train else "")
                  + f"| val deviance: {val_dev:.4f}")

        if should_stop:
            print(f"Early stopping at epoch {epoch}. Best val deviance "
                  f"{stopper.best:.4f} at epoch {stopper.best_epoch}")
            break

    stopper.restore()
    history["best_val_deviance"], history["best_epoch"] = stopper.best, stopper.best_epoch
    if track_train and stopper.best_epoch in history["val_epochs"]:
        history["best_train_deviance"] = history["train_deviance"][history["val_epochs"].index(stopper.best_epoch)]
    return history


class PoissonCANN(nn.Module):
    def __init__(self, in_features, neural_net_params, glm_bias=True, random_state=None):
        """
        Poisson CANN: log(mu) = glm(X) + neural_net(X) [+ log_exposure].
        forward() returns the log-mean (linear predictor), not the mean.
        neural_net_params: dict of MLP params (e.g. hidden layer sizes).
        """
        super().__init__()
        self.glm = LinearPredictor(in_features=in_features, intercept=glm_bias,
                                   random_state=random_state)
        self.neural_net = MLP(in_features=in_features, **neural_net_params,
                              random_state=random_state)

        # Zero the last layer so the CANN starts exactly at the GLM
        torch.nn.init.zeros_(self.neural_net.net[-1].weight)
        torch.nn.init.zeros_(self.neural_net.net[-1].bias)

    def forward(self, X, log_exposure=None):
        eta_glm = self.glm(X)
        eta = eta_glm + self.neural_net(X).view_as(eta_glm)  # avoids (N,) + (N,1) -> (N,N)
        if log_exposure is not None:
            eta = eta + log_exposure.view_as(eta)
        return eta

    @torch.no_grad()
    def predict_mean(self, X, log_exposure=None):
        """Expected claim count mu = exp(eta)."""
        return torch.exp(self.forward(X, log_exposure))

def poisson_mlp_train_loop(model, optimizer, loss_fn, n_epochs, train_loader, val_loader,
                   device="cpu", patience=20, min_delta=1e-9, verbose_every=100):
    model.to(device)

    def unpack(batch):
        # Batches are (X, y) or (X, y, log_exposure)
        X, y, *rest = batch
        log_exp = rest[0].to(device) if rest else None
        return X.to(device), y.to(device), log_exp

    def predict(X, log_exp):
        return model(X) if log_exp is None else model(X, log_exp)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state, best_epoch = None, 0
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        ### Training
        model.train()
        train_loss, n_train = 0.0, 0
        for batch in train_loader:
            X_batch, y_batch, log_exp = unpack(batch)

            y_pred = predict(X_batch, log_exp)
            loss = loss_fn(y_pred=y_pred, y_true=y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            n_train += len(y_batch)
        train_loss /= n_train

        ### Validation
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.inference_mode():
            for batch in val_loader:
                X_batch, y_batch, log_exp = unpack(batch)

                val_pred = predict(X_batch, log_exp)
                batch_loss = loss_fn(y_pred=val_pred, y_true=y_batch)

                val_loss += batch_loss.item() * len(y_batch)
                n_val += len(y_batch)
        val_loss /= n_val

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

def poisson_cann_training_loop(cann_model, glm_loss_function, nn_loss_function,
                       nn_optimiser, nn_optim_params, nn_epochs,
                       train_loader, val_loader, early_stop_params=None,
                       device="cpu", random_state=None):
    set_seed(random_state)
    cann_model.to(device)

    # Assumes dataset tensors are (X, y) or (X, y, log_exposure)
    tensors = [t.to(device) for t in train_loader.dataset.tensors]
    X_all, y_all = tensors[0], tensors[1]
    log_exp_all = tensors[2] if len(tensors) > 2 else None

    with torch.no_grad():
        cann_model.glm.linear.weight.zero_()
        cann_model.glm.linear.bias.fill_(torch.log(y_all.sum() / log_exp_all.exp().sum()))
    # Step 1: fit GLM alone (with exposure offset, bypassing forward)
    glm_optimiser = torch.optim.LBFGS(
        cann_model.glm.parameters(), lr=1.0, max_iter=2000,
        line_search_fn="strong_wolfe")

    def glm_closure():
        glm_optimiser.zero_grad()
        eta = cann_model.glm(X_all)
        if log_exp_all is not None:
            eta = eta + log_exp_all.view_as(eta)
        loss = glm_loss_function(eta, y_all.view_as(eta))
        loss.backward()
        return loss

    glm_optimiser.step(glm_closure)

    # Step 2: freeze GLM
    for p in cann_model.glm.parameters():
        p.requires_grad = False

    # Step 3: train network on residual structure
    optimiser = nn_optimiser(cann_model.neural_net.parameters(), **nn_optim_params)
    history = poisson_mlp_train_loop(cann_model, optimiser, nn_loss_function, nn_epochs,
                             train_loader, val_loader, device=device,
                             **(early_stop_params or {}), verbose_every= 25)
    return history

def poisson_deviance(y_true, y_pred, eps=1e-8):
    """
    y_pred: eta, not actualy y (used y_pred for consistency with other torch.loss instances)
    """
    mu = torch.exp(y_pred.clamp(max=20))      # clamp max to avoid overflow
    log_term = torch.where(
        y_true > 0,
        y_true * (torch.log(y_true.clamp(min=eps)) - y_pred),   # log(y/mu) = log y - eta
        torch.zeros_like(y_true),
    )
    return 2 * (log_term - (y_true - mu)).mean()