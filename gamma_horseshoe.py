"""
Gamma BCANN with a regularised-horseshoe network backbone.

Gamma (claim severity) counterpart of HorseshoePoissonBCANN (horseshoe_prior.py) and
build_guide_horseshoe_bcann (poisson_bcann_vi.py).

    eta = log(y_scale) + x'beta + b0 + f(x; w),   mu = exp(eta)
    y_i ~ Gamma(shape = alpha * w_i, rate = alpha * w_i / mu_i)

    beta, b0 : N(0, glm_prior_scale)
    w        : regularised horseshoe (HorseshoeMLP); biases N(0, 1)
    alpha    : LogNormal(shape_prior_loc, shape_prior_scale), as in GammaBCANN

The model has the same signature and site names as GammaBCANN (plus the shrinkage sites
nn_tau, nn_c2, nn_lam_w_{i}), so training, prediction and NUTS reuse the Gamma functions:
gamma_bcann_svi_training_loop, gamma_posterior_mu, gamma_posterior_shape,
gamma_evaluate_bcann and everything in gamma_bcann_hmc_nuts.

Usage (python gamma_horseshoe.py [--data fremtpl|synthetic]):
    model   = HorseshoeGammaBCANN(48, nn_params, y_scale=gamma_weighted_mean(y_train, w_train))
    glm_mle = gamma_fit_glm_mle(X_train, y_train, w_train, y_scale=model.y_scale)
    guide   = gamma_build_guide_horseshoe_bcann(model, X_train[:256], y_train[:256], w_train[:256],
                                                glm_mle=glm_mle)
    history = gamma_bcann_svi_training_loop(model, guide, train_loader, val_loader, n_epochs=300)
"""
import math
import re

import torch
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import (
    AutoGuideList,
    AutoMultivariateNormal,
    AutoLowRankMultivariateNormal,
    AutoNormal
)
from pyro.infer.autoguide.initialization import init_to_value, init_to_sample
from pyro.nn import PyroModule
from poisson_horseshoe import HorseshoeBayesianCANN
from poisson_bcann_vi import _expose, _is_shrinkage
from gamma_bcann_vi import SHAPE_SITE


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class HorseshoeGammaBCANN(PyroModule):
    """
    Gamma BCANN with a regularised-horseshoe backbone.

        eta = log(y_scale) + x'beta + b0 + f(x; w),   mu = exp(eta)
        y   ~ Gamma(alpha * w, alpha * w / mu)

    `kl_weight` (minibatch KL weight pi_i, e.g. Blundell) scales the log prior of every
    latent site (beta, b0, w, tau, c2, lambda, alpha); the likelihood is left unscaled.
    Pair with a guide scaled by the same weight (`_kl_scaled`), as
    gamma_bcann_svi_training_loop does.

    Args
    ----
    in_features : int
    neural_net_params : dict
        Forwarded to HorseshoeMLP (must contain `hidden_activation_nodes`).
    y_scale : float
        Fixed offset log(y_scale); use the weighted mean claim size of the training set.
    glm_prior_scale : float
        Scale of N(0, s) prior on GLM coefficients.
    tau_0, slab_scale, slab_df : float
        Horseshoe hyperparameters (see HorseshoeBayesianCANN).
    shape_prior_loc, shape_prior_scale : float
        LogNormal prior on the Gamma shape alpha.
    """

    def __init__(
        self,
        in_features,
        neural_net_params,
        y_scale=1.0,
        glm_prior_scale=1.0,
        tau_0=1.0,
        slab_scale=2.0,
        slab_df=4.0,
        shape_prior_loc=0.0,
        shape_prior_scale=1.0,
    ):
        super().__init__()
        self.cann = HorseshoeBayesianCANN(
            in_features=in_features,
            neural_net_params=neural_net_params,
            glm_prior_scale=glm_prior_scale,
            tau_0=tau_0,
            slab_scale=slab_scale,
            slab_df=slab_df,
        )
        self.y_scale = float(y_scale)
        self.shape_prior_loc = shape_prior_loc
        self.shape_prior_scale = shape_prior_scale

    def forward(self, X, y=None, weights=None, kl_weight=1.0):
        offset = math.log(self.y_scale)
        # Latent sites (GLM, NN weights, tau, c2, lambda, alpha) are sampled inside this
        # block, so it scales the prior log-density by kl_weight and nothing else.
        with poutine.scale(scale=kl_weight):
            eta_glm = offset + self.cann.linear_pred(X)
            f_nn = self.cann.neural_net(X)
            shape = pyro.sample(SHAPE_SITE, dist.LogNormal(
                torch.tensor(self.shape_prior_loc, device=X.device),
                torch.tensor(self.shape_prior_scale, device=X.device)))
        eta = eta_glm + f_nn
        mu = torch.exp(eta.clamp(min=-30.0, max=30.0))           # expected claim size
        concentration = shape if weights is None else shape * weights

        pyro.deterministic("eta_glm", eta_glm)
        pyro.deterministic("mu", mu)

        with pyro.plate("data", X.shape[0]):          # no scale: likelihood weight stays 1
            pyro.sample("obs", dist.Gamma(concentration, concentration / mu), obs=y)
        return mu


# --------------------------------------------------------------------------- #
# Guide
# --------------------------------------------------------------------------- #

def gamma_build_guide_horseshoe_bcann(model, X, y, weights=None, glm_mle=None, rank=10,
                                      init_scale=0.05, tau_init=None):
    """
    Guide for HorseshoeGammaBCANN.

    q(beta, b0)         = full-covariance Gaussian          (AutoMultivariateNormal)
    q(alpha)            = log-normal                        (AutoNormal on log alpha)
    q(w)                = low-rank + diagonal Gaussian      (AutoLowRankMultivariateNormal)
    q(tau, c2, lambda)  = independent log-normals (mean-field, AutoNormal)
    q = q(beta, b0) * q(alpha) * q(w) * q(tau, c2, lambda)

    alpha has its own block (as in gamma_build_bcann_guide) so the shape warm-up in
    gamma_bcann_svi_training_loop can hide it without touching the shrinkage scales.

    Initialisation is the same CANN start as gamma_build_bcann_guide (GLM and alpha at the
    MLE / Pearson estimate, final network layer at zero), with the shrinkage scales at their
    prior medians: tau = tau_0 (or `tau_init`), c2 = slab_scale^2, lambda = 1.
    """
    with torch.no_grad():
        tr = poutine.trace(model).get_trace(X, y, weights=weights)
    sites = {n: s["value"] for n, s in tr.nodes.items()
             if s["type"] == "sample" and not s["is_observed"]
             and ("linear_pred" in n or "neural_net" in n or n == SHAPE_SITE
                  or _is_shrinkage(n))}
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
    shape_guide = AutoNormal(
        poutine.block(model, expose_fn=_expose(SHAPE_SITE)),
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
    guide.append(shape_guide)
    guide.append(nn_guide)
    guide.append(shrink_guide)
    guide(X, y, weights=weights)   # run once so the prototype / parameters are created
    return guide
