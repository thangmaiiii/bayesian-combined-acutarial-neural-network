"""
Gamma BCANN: HMC-NUTS implementation  (CUDA-aware)

Gamma (claim severity) counterpart of poisson_bcann_hmc_nuts.py. The model is
gamma_bcann_vi.GammaBCANN; every function takes the claim-count weights `weights`
where the Poisson version takes `log_exposure`.

Device helpers (get_device, to_device) and width_by_bin are re-used from
poisson_bcann_hmc_nuts.

Usage (python gamma_bcann_hmc_nuts.py [--data fremtpl|synthetic]):
    device = get_device()
    sub    = gamma_stratified_subsample(y_train, n=5000)
    Xs, ys, ws = X_train[sub], y_train[sub], w_train[sub]
    inits  = gamma_make_chain_inits(model, Xs, ys, ws, num_chains=4, glm_mle=glm_mle,
                              guide=guide, how="svi_sample", shape_init="pearson",
                              device=device)
    samples, info = gamma_run_nuts(model, Xs, ys, ws, inits, device=device)
    gamma_nuts_diagnostics(model, samples, info, X_val, w_val, y_val, device=device)
"""

import os, re, time
import torch, pyro
from pyro import poutine
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.autoguide.initialization import init_to_value, init_to_sample
from codebase_0 import set_seed
from gamma_bcann_vi import SHAPE_SITE, gamma_deviance
from poisson_bcann_hmc_nuts import (
    get_device, to_device, width_by_bin, _empty_cache, _flat, _Running, _rhat_ess
)


# ---------------------------------------------------------------- data helpers
def gamma_stratified_subsample(y, n, n_bins=10, seed=0):
    """Indices of ~n rows, keeping the share of each claim-size quantile bin
    (so the large-claim tail is represented in the subsample).

    Returned indices live on the same device as `y`.
    """
    src_device = y.device
    y_cpu = y.cpu()
    g = torch.Generator().manual_seed(seed)
    frac = n / y_cpu.numel()
    edges = torch.quantile(y_cpu, torch.linspace(0, 1, n_bins + 1)[1:-1])
    bins = torch.bucketize(y_cpu, edges)
    pick = lambda idx: idx[torch.randperm(idx.numel(), generator=g)[: round(frac * idx.numel())]]
    idx = torch.cat([pick((bins == b).nonzero().squeeze(-1)) for b in range(n_bins)]).sort().values
    return idx.to(src_device)


# ---------------------------------------------------------------- initialisation
def _gamma_latent_values(model, X, y, weights):
    with torch.no_grad():
        tr = poutine.trace(model).get_trace(X, y, weights=weights)
    return {n: s["value"].detach().clone() for n, s in tr.nodes.items()
            if s["type"] == "sample" and not s["is_observed"]}


def gamma_make_chain_inits(model, X, y, weights, num_chains, glm_mle=None, guide=None,
                     how="svi_sample", shape_init="pearson", seed=0, device=None):
    """Build one init dict per chain.

    how : "svi_sample" | "svi_median" | "glm"
        "glm" puts beta, b0 at the GLM MLE, the last network layer at zero and
        other layers at a prior draw.
    shape_init : "pearson" | None
        "pearson" starts the shape in every chain at the GLM Pearson estimate
        glm_mle[2] (whatever `how` is); NUTS then samples it jointly with the
        weights. None takes the shape from `how` instead (an SVI draw / median,
        or a prior draw for "glm").
    device : torch.device or None
        When given, every init tensor is moved to this device.
    """
    if shape_init == "pearson" and (glm_mle is None or len(glm_mle) < 3):
        raise ValueError('shape_init="pearson" needs glm_mle = (weight, bias, shape)')
    names = list(_gamma_latent_values(model, X, y, weights).keys())
    inits = []
    for c in range(num_chains):
        set_seed(seed + c)
        if how == "svi_sample":
            with torch.no_grad():
                tr = poutine.trace(guide).get_trace(X, y, weights=weights)
            vals = {n: tr.nodes[n]["value"].detach().clone() for n in names if n in tr.nodes}
        elif how == "svi_median":
            with torch.no_grad():
                med = guide.median(X, y, weights=weights)
            vals = {n: med[n].detach().clone() for n in names if n in med}
        elif how == "glm":
            vals = _gamma_latent_values(model, X, y, weights)
            last = max(int(m.group(1)) for n in vals
                       if (m := re.search(r"neural_net\.net\.(\d+)\.", n)))
            for n in list(vals):
                if f"neural_net.net.{last}." in n:
                    vals[n] = torch.zeros_like(vals[n])
                if glm_mle is not None and "linear_pred" in n:
                    vals[n] = (glm_mle[0] if n.endswith("weight") else glm_mle[1]).clone()
        else:
            raise ValueError(how)

        if shape_init == "pearson":
            vals[SHAPE_SITE] = glm_mle[2].clone().reshape(vals[SHAPE_SITE].shape)
        elif shape_init is not None:
            raise ValueError(shape_init)

        if device is not None:
            vals = {n: v.to(device) for n, v in vals.items()}

        inits.append(vals)
    return inits


# ---------------------------------------------------------------- sampling
def gamma_run_nuts(model, X, y, weights, inits, num_warmup=500, num_samples=500,
             target_accept=0.8, max_tree_depth=8, full_mass_glm=True, seed=0,
             progress_bar=True, device=None, jit_compile=False):
    """One sequential NUTS chain per entry of `inits`.

    The GLM block (beta, b0) gets a dense mass matrix; the shape and the
    network weights get diagonal ones. The shape is sampled jointly with the
    weights (on the log scale), starting from its value in `inits`. Samples
    are returned on CPU.
    """
    if device is not None:
        model = to_device(device, model)
        X, y, weights = to_device(device, X, y, weights)

    names = list(inits[0].keys())
    glm_names = tuple(n for n in names if "linear_pred" in n)
    full_mass = [glm_names] if (full_mass_glm and glm_names) else False

    per_chain, info = [], {"divergences": [], "step_size": [], "seconds": [], "device": str(X.device)}
    for c, init in enumerate(inits):
        _empty_cache(device)

        if device is not None:
            init = {n: v.to(device) for n, v in init.items()}

        pyro.set_rng_seed(seed + c)
        kernel = NUTS(model, target_accept_prob=target_accept, max_tree_depth=max_tree_depth,
                      full_mass=full_mass,
                      init_strategy=init_to_value(values=init, fallback=init_to_sample),
                      jit_compile=jit_compile, ignore_jit_warnings=True)
        mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=num_warmup,
                    num_chains=1, disable_progbar=not progress_bar)
        t0 = time.time()
        mcmc.run(X, y, weights=weights)
        info["seconds"].append(time.time() - t0)

        if num_samples >= 4:
            info["divergences"].append(len(mcmc.diagnostics()["divergences"]["chain 0"]))
        else:
            info["divergences"].append(0)

        info["step_size"].append(float(kernel.step_size))
        per_chain.append(mcmc.get_samples())

    samples = {}
    for k in per_chain[0]:
        stacked = torch.stack([ch[k] for ch in per_chain])
        samples[k] = stacked.cpu() if (device is not None and device.type != "cpu") else stacked

    _empty_cache(device)
    return samples, info


def gamma_time_nuts(model, X, y, weights, inits, n_iter=20, device=None, jit_compile=False):
    """Seconds per NUTS iteration (early warmup) on the specified device."""
    t0 = time.time()
    gamma_run_nuts(model, X, y, weights, inits[:1], num_warmup=n_iter, num_samples=1,
             progress_bar=False, device=device, jit_compile=jit_compile)
    s = (time.time() - t0) / n_iter
    dev_str = str(device) if device else "current"
    print(f"{s:.2f} s / iteration on {dev_str}"
          f"  (early warmup iterations are not representative of tree depth)")
    return s


# ---------------------------------------------------------------- prediction
@torch.no_grad()
def gamma_predict_from_samples(model, samples, X, batch=250, device=None):
    """For SMALL X (diagnostic subsets).

    Returns mu, eta_glm, f_nn as [chains, draws, n] and shape as [chains, draws, 1].
    Results are on CPU.
    """
    if device is not None:
        model = to_device(device, model)
        X = to_device(device, X)

    flat, C, D = _flat(samples)
    sites = ("mu", "eta_glm")
    out = {s: [] for s in sites}
    for i in range(0, C * D, batch):
        chunk = {k: v[i:i + batch] for k, v in flat.items()}
        if device is not None:
            chunk = {k: v.to(device) for k, v in chunk.items()}

        pred = Predictive(model, posterior_samples=chunk, return_sites=sites)(X)
        for s in sites:
            out[s].append(pred[s].cpu() if device is not None else pred[s])

        _empty_cache(device)

    out = {s: torch.cat(v).reshape(C, D, -1) for s, v in out.items()}
    out["f_nn"] = torch.log(out["mu"]) - out["eta_glm"]
    out["shape"] = samples[SHAPE_SITE].cpu().reshape(C, D, 1)
    return out


@torch.no_grad()
def gamma_nuts_posterior_stats(model, samples, X, batch=250, device=None):
    """Posterior mean of mu and mean / sd of log(mu) per row of X (streamed)."""
    if device is not None:
        model = to_device(device, model)
        X = to_device(device, X)

    flat, C, D = _flat(samples)
    acc = _Running()
    for i in range(0, C * D, batch):
        chunk = {k: v[i:i + batch] for k, v in flat.items()}
        if device is not None:
            chunk = {k: v.to(device) for k, v in chunk.items()}

        acc.add(Predictive(model, posterior_samples=chunk, return_sites=("mu",))(X)["mu"])

        _empty_cache(device)

    return acc.result()


@torch.no_grad()
def gamma_svi_posterior_stats(model, guide, X, num_samples=500, batch=100, device=None):
    if device is not None:
        model, guide = to_device(device, model, guide)
        X = to_device(device, X)

    acc = _Running()
    for i in range(0, num_samples, batch):
        b = min(batch, num_samples - i)
        acc.add(Predictive(model, guide=guide, num_samples=b, return_sites=("mu",))(X)["mu"])
        _empty_cache(device)
    return acc.result()


def gamma_deviance_from_stats(stats, y, weights=None):
    """Weighted Gamma deviance of the posterior-mean claim size."""
    return gamma_deviance(
        y_true=y.cpu(),
        y_pred=torch.log(stats["mean_mu"].cpu()),
        weights=None if weights is None else weights.cpu(),
    ).item()


# ---------------------------------------------------------------- diagnostics
@torch.no_grad()
def gamma_nuts_diagnostics(model, samples, info, X_eval, weights_eval=None, y_eval=None,
                     n_points=200, seed=0, device=None):
    """R-hat / ESS on identifiable functions of the weights, the GLM block and the shape."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X_eval.shape[0], generator=g)[:n_points].to(X_eval.device)
    pred = gamma_predict_from_samples(model, samples, X_eval[idx], device=device)

    rows = {k: _rhat_ess(pred[k]) for k in ("mu", "eta_glm", "f_nn", "shape")}
    for n, v in samples.items():
        if "linear_pred" in n:
            rows[n] = _rhat_ess(v.reshape(v.shape[0], v.shape[1], -1))
    if y_eval is not None:
        w = torch.ones(idx.numel()) if weights_eval is None else weights_eval[idx].cpu()
        conc = pred["shape"] * w
        ll = torch.distributions.Gamma(conc, conc / pred["mu"]).log_prob(
            y_eval[idx].cpu()).sum(-1, keepdim=True)
        rows["loglik (eval subset)"] = _rhat_ess(ll)

    shape = pred["shape"].reshape(-1)
    dev_str = info.get("device", "unknown")
    print(f"Device used             : {dev_str}")
    print(f"Wall time per chain (s) : {[round(s, 1) for s in info['seconds']]}")
    print(f"Divergences per chain   : {info['divergences']}")
    print(f"Adapted step size       : {[round(s, 5) for s in info['step_size']]}")
    print(f"Gamma shape (posterior) : mean {shape.mean():.3f}, "
          f"95% CI [{shape.quantile(0.025):.3f}, {shape.quantile(0.975):.3f}]")
    print(f"{'quantity':<40}{'max R-hat':>10}{'min ESS':>10}")
    for k, (rh, ess) in rows.items():
        print(f"{k:<40}{rh.max().item():>10.3f}{ess.min().item():>10.0f}")
    return rows