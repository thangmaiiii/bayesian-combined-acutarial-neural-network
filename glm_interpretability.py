"""
Does the GLM block of the BCANN (Poisson frequency or Gamma severity) keep its actuarial meaning?

Compares beta from
  (i)   the standalone GLM (MLE, Fisher-information standard errors),
  (ii)  the VI-fitted BCANN (q(beta, b0) from runs/{poisson,gamma}_bcann_vi.pt),
  (iii) the HMC posterior (hmc_results/{poisson,gamma}_hmcnuts_full_set.pt),
and measures how much linear signal the network f(x; w) has absorbed from the GLM.
Both families use a log link, so everything below applies to either; only the GLM fit
(glm_mle_with_se vs gamma_glm_mle_with_se) and the absorption weights differ.

All three were fitted on X_learn (learn_scaler), so the coefficients are on the same
MinMax scale: a dummy's beta is its log-relativity, a continuous covariate's beta is the
log-relativity across its full observed range.

Coefficient vectors are ordered [intercept, w_1, ..., w_p] throughout.
"""
import numpy as np
import pandas as pd
import torch
import pyro
from torch import nn
from pyro.infer import Predictive

from poisson_bcann_vi import PoissonBCANN, build_bcann_guide, poisson_fit_glm_mle
from gamma_bcann_vi import GammaBCANN, gamma_build_bcann_guide, gamma_fit_glm_mle

W_SITE, B_SITE = "cann.linear_pred.layer.weight", "cann.linear_pred.layer.bias"

# recipe["cls"] -> (model class, guide builder name, guide builder)
_RUN_TYPES = {
    "PoissonBCANN": (PoissonBCANN, "build_bcann_guide", build_bcann_guide),
    "GammaBCANN": (GammaBCANN, "gamma_build_bcann_guide", gamma_build_bcann_guide),
}


def _design(X):
    """[1, X] in float64."""
    X = X.double()
    return torch.cat([torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device), X], dim=1)


# --------------------------------------------------------------------------- #
# Coefficients from each fit
# --------------------------------------------------------------------------- #

def glm_mle_with_se(X, y, log_exposure, random_state=None):
    """
    Standalone GLM: MLE from poisson_fit_glm_mle plus the inverse Fisher information
    (X~' diag(mu) X~)^{-1}, the usual GLM covariance with canonical log link.
    Returns (theta [p+1], cov [p+1, p+1], mu [n]).
    """
    w, b = poisson_fit_glm_mle(X, y, log_exposure=log_exposure, random_state=random_state)
    theta = torch.cat([b.reshape(1), w.reshape(-1)]).double()
    Xd = _design(X)
    mu = torch.exp(Xd @ theta + log_exposure.double())
    info = Xd.T @ (Xd * mu[:, None])
    return theta, torch.linalg.inv(info), mu


def gamma_glm_mle_with_se(X, y, weights=None, y_scale=1.0, random_state=None):
    """
    Standalone Gamma GLM (log link): MLE from gamma_fit_glm_mle plus the inverse Fisher
    information (alpha X~' diag(w) X~)^{-1}, alpha the Pearson shape. With a log link the
    Gamma working weight is alpha * w_i, free of mu.
    Returns (theta [p+1], cov [p+1, p+1], mu [n]); theta excludes the log(y_scale) offset,
    as the BCANN's b0 does.
    """
    w_, b, shape = gamma_fit_glm_mle(X, y, weights=weights, y_scale=y_scale,
                                     random_state=random_state)
    theta = torch.cat([b.reshape(1), w_.reshape(-1)]).double().cpu()
    Xd = _design(X.cpu())
    w = torch.ones(Xd.shape[0], dtype=Xd.dtype) if weights is None else weights.double().cpu()
    mu = torch.exp(Xd @ theta + np.log(y_scale))
    info = shape.double().cpu() * (Xd.T @ (Xd * w[:, None]))
    return theta, torch.linalg.inv(info), mu


def load_vi(run_path, X, y, device="cpu"):
    """Rebuild a PoissonBCANN / GammaBCANN and its guide from a runs/*.pt file saved by the VI notebook."""
    run = torch.load(run_path, map_location=device, weights_only=False)
    cls, builder_name, builder = _RUN_TYPES[run["recipe"]["cls"]]
    assert run["guide_builder"] == builder_name, run["guide_builder"]
    kwargs = dict(run["recipe"]["kwargs"])
    nn_params = dict(kwargs["neural_net_params"])
    if isinstance(nn_params.get("hidden_activations"), str):    # saved as e.g. "Tanh"
        nn_params["hidden_activations"] = getattr(nn, nn_params["hidden_activations"])
    kwargs["neural_net_params"] = nn_params
    pyro.clear_param_store()
    model = cls(**kwargs).to(device)
    guide = builder(model, X[:256], y[:256], **run["guide_kwargs"])
    guide.load_state_dict(run["guide_state"])
    return model, guide


@torch.no_grad()
def vi_glm_draws(model, guide, X, num_samples=4000):
    """Draws of [b0, beta] from q, shape [num_samples, p+1]."""
    pred = Predictive(model, guide=guide, num_samples=num_samples,
                      return_sites=(W_SITE, B_SITE))(X[:2])
    return torch.cat([pred[B_SITE].reshape(num_samples, 1),
                      pred[W_SITE].reshape(num_samples, -1)], dim=1).double()


def hmc_glm_draws(samples, thin=1):
    """Draws of [b0, beta] pooled over chains, shape [chains * draws / thin, p+1]."""
    b = samples[B_SITE][:, ::thin].reshape(-1, 1)
    w = samples[W_SITE][:, ::thin]
    return torch.cat([b, w.reshape(b.shape[0], -1)], dim=1).double()


# --------------------------------------------------------------------------- #
# Network output per posterior draw
# --------------------------------------------------------------------------- #

@torch.no_grad()
def vi_f_nn(model, guide, X, num_samples=1000, batch=100):
    """
    f(x; w) and [b0, beta] for the SAME draws from q: f is [num_samples, n].
    Pairing matters: the absorption check compares beta^(s) with the network of draw s.
    """
    f, theta = [], []
    for i in range(0, num_samples, batch):
        k = min(batch, num_samples - i)
        pred = Predictive(model, guide=guide, num_samples=k,
                          return_sites=("eta_glm", "mu", W_SITE, B_SITE))(X)
        f.append(torch.log(pred["mu"]).reshape(k, -1) - pred["eta_glm"].reshape(k, -1))
        theta.append(torch.cat([pred[B_SITE].reshape(k, 1), pred[W_SITE].reshape(k, -1)], 1))
    return torch.cat(f).double(), torch.cat(theta).double()


@torch.no_grad()
def hmc_f_nn(model, samples, X, thin=5, batch=250):
    """f(x; w) and [b0, beta] for the HMC draws (thinned), f is [draws, n]."""
    thinned = {k: v[:, ::thin] for k, v in samples.items()}
    flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in thinned.items()}
    S = flat[B_SITE].shape[0]
    f = []
    for i in range(0, S, batch):
        chunk = {k: v[i:i + batch] for k, v in flat.items()}
        pred = Predictive(model, posterior_samples=chunk, return_sites=("eta_glm", "mu"))(X)
        f.append(torch.log(pred["mu"]).reshape(len(chunk[B_SITE]), -1)
                 - pred["eta_glm"].reshape(len(chunk[B_SITE]), -1))
    return torch.cat(f).double(), hmc_glm_draws(samples, thin)


# --------------------------------------------------------------------------- #
# Step 5: how much of the GLM has the network absorbed?
# --------------------------------------------------------------------------- #

def nn_absorption(f, theta, X, weights):
    """
    Project each draw of the network output onto the GLM design in the GLM's
    information metric W = diag(weights): exposure * mu_GLM for Poisson, the claim-count
    weights w for Gamma (its log-link working weight alpha * w; the constant alpha cancels):

        gamma^(s) = (X~' W X~)^{-1} X~' W f^(s)

    gamma is the part of f that a GLM can express: refitting the standalone GLM with f
    as an extra offset moves its MLE by about -gamma. So if the network has absorbed
    main effects, beta_BCANN ≈ beta_GLM - gamma, and beta_eff = beta + gamma is the
    coefficient the BCANN actually implies for the linear part of eta.

    f      : [S, n] network output per draw
    theta  : [S, p+1] matching [b0, beta] draws
    X      : [n, p] covariates (same rows as f)
    weights: [n] exposure * mu_GLM (Poisson) or w (Gamma)

    Returns dict of tensors:
      gamma [S, p+1], beta_eff [S, p+1],
      r2_linear [S]   weighted R^2 of f on X~: share of the network's variation that is
                      linear in X (absorbed main effects) vs interactions / non-linearity,
      var_share [S]   Var_W(f) / Var_W(eta) with eta = X~ beta + f.
    """
    Xd = _design(X)
    w = weights.double()
    XtW = Xd.T * w
    gamma = torch.linalg.solve(XtW @ Xd, XtW @ f.T).T          # [S, p+1]

    def wvar(a):                                                # weighted variance per draw
        m = (a * w).sum(1, keepdim=True) / w.sum()
        return ((a - m) ** 2 * w).sum(1) / w.sum()

    resid = f - gamma @ Xd.T
    eta = theta @ Xd.T + f
    return {
        "gamma": gamma,
        "beta_eff": theta + gamma,
        "r2_linear": 1 - wvar(resid) / wvar(f),
        "var_share": wvar(f) / wvar(eta),
    }


def absorption_table(names, theta_glm, theta_bcann, absorb):
    """
    Per coefficient: observed drift from the GLM, drift predicted by absorption (-gamma),
    and the drift left once the absorbed part is added back (beta_eff - beta_GLM).
    """
    drift = theta_bcann.mean(0) - theta_glm
    return pd.DataFrame({
        "drift": drift.numpy(),
        "minus_gamma": (-absorb["gamma"].mean(0)).numpy(),
        "drift_after_absorption": (absorb["beta_eff"].mean(0) - theta_glm).numpy(),
        # draws where beta_j and gamma_j trade off show up as strong negative correlation
        "corr_beta_gamma": [np.corrcoef(theta_bcann[:, j], absorb["gamma"][:, j])[0, 1]
                            for j in range(theta_glm.numel())],
    }, index=names)


# --------------------------------------------------------------------------- #
# Are the coefficients different?
# --------------------------------------------------------------------------- #

def compare_coefficients(names, theta_glm, se_glm, draws, materiality=0.05, alpha=0.05):
    """
    Compare a posterior over [b0, beta] with the GLM MLE, coefficient by coefficient.

    Both fits use the same data, so the question is not "do two noisy estimates differ"
    but "did the BCANN move beta further than the GLM's own estimation error, and does
    the move matter for pricing". Hence two separate checks:

    shift_in_glm_se  (posterior mean - MLE) / SE_GLM: the move in units of GLM noise.
    tail_prob        2 * min(P(beta_j <= MLE_j), P(beta_j >= MLE_j)): where the MLE sits in
                     the BCANN posterior. < alpha  => statistically distinguishable.
    rel_ratio        exp(posterior mean - MLE): multiplicative change in the tariff
                     relativity (1.08 => 8% higher than the GLM's).
                     |rel_ratio - 1| > materiality => material for rating.
    sd_ratio         posterior sd / SE_GLM: does the BCANN carry the same uncertainty?

    verdict: 'same' (neither), 'different' (both), 'detectable, immaterial',
             'material, not detectable'.
    """
    d = draws - theta_glm
    shift = draws.mean(0) - theta_glm
    rel_ratio = torch.exp(shift)
    tail = 2 * torch.minimum((d <= 0).double().mean(0), (d >= 0).double().mean(0))
    lo, hi = torch.quantile(draws, torch.tensor([0.025, 0.975], dtype=draws.dtype), dim=0)

    detectable = (tail < alpha).numpy()
    material = ((rel_ratio - 1).abs() > materiality).numpy()
    verdict = np.select([detectable & material, detectable, material],
                        ["different", "detectable, immaterial", "material, not detectable"],
                        default="same")
    return pd.DataFrame({
        "glm": theta_glm.numpy(), "glm_se": se_glm.numpy(),
        "post_mean": draws.mean(0).numpy(), "post_sd": draws.std(0).numpy(),
        "lo95": lo.numpy(), "hi95": hi.numpy(),
        "shift_in_glm_se": (shift / se_glm).numpy(),
        "rel_ratio": rel_ratio.numpy(),
        "sd_ratio": (draws.std(0) / se_glm).numpy(),
        "tail_prob": tail.numpy(),
        "verdict": verdict,
    }, index=names)


def relativity_ratio_table(names, theta_glm, se_glm, sources, drop_intercept=True):
    """
    Relativity ratio against the standalone GLM, for the GLM itself and each posterior:

        r_j = exp(mu_j - beta_hat_j)

    with mu_j the posterior mean (the MLE for the GLM, so r_j = 1). Multiplicative change
    in the tariff relativity: r_j = 1.08 => that factor's relativity is 8% above the GLM's.
    Unlike (mu_j - beta_hat_j) / beta_hat_j it stays finite when beta_hat_j ≈ 0.

    95% intervals are exp(quantiles of beta^(s) - beta_hat) for the posteriors and
    exp(±1.96 SE_GLM) for the GLM, so the GLM row is the band its own estimation noise
    allows: a posterior ratio inside it has moved no more than the GLM could by chance.

    sources: {label: draws [S, p+1]}, e.g. {"VI": theta_vi, "HMC": theta_hmc}.
    Returns a DataFrame with columns (label, {relativity, r, r_lo95, r_hi95}).
    """
    j0 = 1 if drop_intercept else 0
    q = lambda dr: torch.tensor([0.025, 0.975], dtype=dr.dtype)
    cols = {
        ("GLM", "relativity"): torch.exp(theta_glm[j0:]),
        ("GLM", "r"): torch.ones_like(theta_glm[j0:]),
        ("GLM", "r_lo95"): torch.exp(-1.96 * se_glm[j0:]),
        ("GLM", "r_hi95"): torch.exp(1.96 * se_glm[j0:]),
    }
    for lab, dr in sources.items():
        d = dr[:, j0:] - theta_glm[j0:]
        lo, hi = torch.quantile(d, q(d), dim=0)
        cols[(lab, "relativity")] = torch.exp(dr[:, j0:].mean(0))
        cols[(lab, "r")] = torch.exp(d.mean(0))
        cols[(lab, "r_lo95")] = torch.exp(lo)
        cols[(lab, "r_hi95")] = torch.exp(hi)
    return pd.DataFrame({k: v.numpy() for k, v in cols.items()}, index=list(names)[j0:])


def variance_comparison(names, draws_vi, draws_hmc, drop_intercept=True):
    """
    Does VI get the posterior spread of [b0, beta] right? HMC is the reference.

    Per coefficient (DataFrame):
      sd_ratio   sd_VI / sd_HMC; < 1 => VI overconfident about that factor (the usual
                 KL(q || p) direction).
    Jointly (dict), from the eigenvalues lambda of Sigma_HMC^{-1} Sigma_VI (all 1 if equal):
      sd_ratio_min_dir    sqrt(min lambda): worst-case sd ratio over any linear combination
      sd_ratio_max_dir    sqrt(max lambda)
      mean_log_var_ratio  (1/p) log(det Sigma_VI / det Sigma_HMC); < 0 => VI too narrow overall
      max_abs_corr_diff   largest gap between the two correlation matrices: did q capture
                          the trade-offs between coefficients?

    The HMC sd carries Monte Carlo error of roughly sd / sqrt(2 ESS); read sd_ratio with that
    in mind.
    """
    j0 = 1 if drop_intercept else 0
    a, b = draws_vi[:, j0:], draws_hmc[:, j0:]
    sd_vi, sd_hmc = a.std(0), b.std(0)
    per_coef = pd.DataFrame({
        "sd_vi": sd_vi.numpy(), "sd_hmc": sd_hmc.numpy(),
        "sd_ratio": (sd_vi / sd_hmc).numpy(),
    }, index=list(names)[j0:])

    S_vi, S_hmc = torch.cov(a.T), torch.cov(b.T)
    lam = torch.linalg.eigvals(torch.linalg.solve(S_hmc, S_vi)).real
    C_vi = S_vi / torch.outer(sd_vi, sd_vi)
    C_hmc = S_hmc / torch.outer(sd_hmc, sd_hmc)
    joint = {
        "sd_ratio_min_dir": lam.min().sqrt().item(),
        "sd_ratio_max_dir": lam.max().sqrt().item(),
        "mean_log_var_ratio": lam.log().mean().item(),
        "max_abs_corr_diff": (C_vi - C_hmc).abs().max().item(),
    }
    return per_coef, joint


def joint_tail_prob(theta_glm, draws, exclude_intercept=True):
    """
    Joint check: Mahalanobis distance of the GLM estimate from the posterior mean, in the
    posterior's own covariance, and the share of draws at least that far out.
    Near 0 => the GLM coefficient vector as a whole is implausible under the BCANN posterior.
    """
    sl = slice(1, None) if exclude_intercept else slice(None)
    D = draws[:, sl]
    m, P = D.mean(0), torch.linalg.inv(torch.cov(D.T))
    md = lambda v: ((v - m) @ P * (v - m)).sum(-1)
    d_glm = md(theta_glm[sl])
    return {"mahalanobis2": d_glm.item(), "df": D.shape[1],
            "tail_prob": (md(D) >= d_glm).double().mean().item()}


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def forest_plot(names, theta_glm, se_glm, sources, ax=None, drop_intercept=True):
    """
    Relativities exp(beta) with 95% intervals.
    sources: {label: draws [S, p+1]}; the GLM uses MLE ± 1.96 SE.
    """
    import matplotlib.pyplot as plt
    j0 = 1 if drop_intercept else 0
    labels = list(names)[j0:]
    k = len(sources) + 1
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.3 * len(labels) * k / 2 + 1.5))
    y = np.arange(len(labels))
    off = np.linspace(-0.3, 0.3, k)

    rows = [("GLM", theta_glm[j0:], theta_glm[j0:] - 1.96 * se_glm[j0:],
             theta_glm[j0:] + 1.96 * se_glm[j0:])]
    for lab, dr in sources.items():
        lo, hi = torch.quantile(dr[:, j0:], torch.tensor([0.025, 0.975], dtype=dr.dtype), dim=0)
        rows.append((lab, dr[:, j0:].mean(0), lo, hi))

    for o, (lab, m, lo, hi) in zip(off, rows):
        m, lo, hi = np.exp(m.numpy()), np.exp(lo.numpy()), np.exp(hi.numpy())
        ax.errorbar(m, y + o, xerr=[m - lo, hi - m], fmt="o", ms=3, capsize=2, label=lab)
    ax.axvline(1, ls="--", c="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Relativity exp(β)")
    ax.legend()
    return ax


def absorption_plot(names, tab, ax=None, label_top=8):
    """Observed drift vs drift predicted by absorption (-gamma); points on y = x => explained."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    t = tab.drop(index=names[0])                                # drop intercept
    ax.scatter(t["minus_gamma"], t["drift"], s=14)
    lim = np.abs(t[["drift", "minus_gamma"]].to_numpy()).max() * 1.1
    ax.plot([-lim, lim], [-lim, lim], ls="--", c="grey", lw=0.8)
    for n in t["drift"].abs().nlargest(label_top).index:
        ax.annotate(n, (t.loc[n, "minus_gamma"], t.loc[n, "drift"]), fontsize=7)
    ax.set_xlabel("Drift predicted by absorption  (−γ)")
    ax.set_ylabel("Observed drift  (β_BCANN − β_GLM)")
    return ax
