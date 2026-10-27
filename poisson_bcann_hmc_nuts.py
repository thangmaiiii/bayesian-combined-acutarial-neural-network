"""
Poisson BCANN: model, guide and HMC-NUTS implementation  (CUDA-aware)

Changes from codebase_1:
  • get_device()          – pick CPU / CUDA / MPS with one call
  • to_device()           – move model + arbitrary tensors in one call
  • stratified_subsample  – returns indices on the source tensor's device
  • make_chain_inits      – moves every init value to `device` when given
  • run_nuts              – accepts `device`, empties CUDA cache between chains
  • predict_from_samples  – respects device of X (no change needed)
  • nuts_diagnostics      – prints device used

Usage:
    device = get_device()                         # "cuda" if available
    model  = to_device(device, model)             # move model
    X, y, log_exp = to_device(device, X, y, log_exp)   # move data
    inits  = make_chain_inits(..., device=device)
    samples, info = run_nuts(..., device=device)
"""

import re, time
import torch, pyro
from pyro import poutine
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.autoguide.initialization import init_to_value, init_to_sample
from pyro.ops.stats import effective_sample_size, split_gelman_rubin
from codebase_0 import poisson_deviance, set_seed


# ---------------------------------------------------------------- device helpers
def get_device(prefer="cuda"):
    """Return the best available torch device.

    prefer : "cuda" | "mps" | "cpu"
        Tries the preferred backend first, then falls back through
        cuda → mps → cpu.
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer != "cpu":
        # fallback chain: try cuda, then mps, then cpu
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def to_device(device, *args):
    """Move every arg (Module or Tensor) to `device`.

    Returns a single object when one arg is passed, otherwise a tuple.

        model = to_device(device, model)
        X, y, log_exp = to_device(device, X, y, log_exp)
    """
    moved = []
    for obj in args:
        if isinstance(obj, torch.nn.Module):
            moved.append(obj.to(device))
        elif isinstance(obj, torch.Tensor):
            moved.append(obj.to(device))
        else:
            moved.append(obj)          # pass through non-torch objects unchanged
    return moved[0] if len(moved) == 1 else tuple(moved)


def _empty_cache(device):
    """Free cached GPU memory between chains to prevent OOM."""
    if device is not None and device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- data helpers
def stratified_subsample(y, n, seed=0):
    """Indices of ~n rows, keeping the claim / no-claim ratio.

    Returned indices live on the same device as `y` so they can be used
    directly for indexing without an extra .to() call.
    """
    src_device = y.device
    y_cpu = y.cpu()
    g = torch.Generator().manual_seed(seed)
    frac = n / y_cpu.numel()
    pick = lambda idx: idx[torch.randperm(idx.numel(), generator=g, device="cpu")[: round(frac * idx.numel())]]
    idx = torch.cat([
        pick((y_cpu > 0).nonzero().squeeze(-1)),
        pick((y_cpu == 0).nonzero().squeeze(-1)),
    ]).sort().values
    return idx.to(src_device)


# ---------------------------------------------------------------- initialisation
def _latent_values(model, X, y, log_exposure):
    with torch.no_grad():
        tr = poutine.trace(model).get_trace(X, y, log_exposure=log_exposure)
    return {n: s["value"].detach().clone() for n, s in tr.nodes.items()
            if s["type"] == "sample" and not s["is_observed"]}


def make_chain_inits(model, X, y, log_exposure, num_chains, glm_mle=None, guide=None,
                     how="svi_sample", seed=0, device=None):
    """Build one init dict per chain.

    how : "svi_sample" | "svi_median" | "glm"
    device : torch.device or None
        When given, every init tensor is moved to this device so it
        matches the device used by run_nuts.
    """
    names = list(_latent_values(model, X, y, log_exposure).keys())
    inits = []
    for c in range(num_chains):
        set_seed(seed + c)
        if how == "svi_sample":
            with torch.no_grad():
                tr = poutine.trace(guide).get_trace(X, y, log_exposure=log_exposure)
            vals = {n: tr.nodes[n]["value"].detach().clone() for n in names if n in tr.nodes}
        elif how == "svi_median":
            with torch.no_grad():
                med = guide.median(X, y, log_exposure=log_exposure)
            vals = {n: med[n].detach().clone() for n in names if n in med}
        elif how == "glm":
            vals = _latent_values(model, X, y, log_exposure)
            last = max(int(m.group(1)) for n in vals
                       if (m := re.search(r"neural_net\.net\.(\d+)\.", n)))
            for n in list(vals):
                if f"neural_net.net.{last}." in n:
                    vals[n] = torch.zeros_like(vals[n])
                if glm_mle is not None and "linear_pred" in n:
                    vals[n] = (glm_mle[0] if n.endswith("weight") else glm_mle[1]).clone()
        else:
            raise ValueError(how)

        # move every init value to the target device
        if device is not None:
            vals = {n: v.to(device) for n, v in vals.items()}

        inits.append(vals)
    return inits


# ---------------------------------------------------------------- sampling
def run_nuts(model, X, y, log_exposure, inits, num_warmup=500, num_samples=500,
             target_accept=0.8, max_tree_depth=8, full_mass_glm=True, seed=0,
             progress_bar=True, device=None, jit_compile=False):
    """One sequential NUTS chain per entry of `inits`.

    device : torch.device or None
        When given, the model and data tensors are moved to this device
        before sampling starts, and the returned samples are moved back
        to CPU so downstream analysis doesn't require GPU memory.
    """
    # ---- optionally relocate model + data ----
    if device is not None:
        model = to_device(device, model)
        X, y, log_exposure = to_device(device, X, y, log_exposure)

    names = list(inits[0].keys())
    glm_names = tuple(n for n in names if "linear_pred" in n)
    full_mass = [glm_names] if (full_mass_glm and glm_names) else False

    per_chain, info = [], {"divergences": [], "step_size": [], "seconds": [], "device": str(X.device)}
    for c, init in enumerate(inits):
        _empty_cache(device)

        # ensure init values are on the correct device
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
        mcmc.run(X, y, log_exposure=log_exposure)
        info["seconds"].append(time.time() - t0)

        if num_samples >= 4:
            info["divergences"].append(len(mcmc.diagnostics()["divergences"]["chain 0"]))
        else:
            info["divergences"].append(0)

        info["step_size"].append(float(kernel.step_size))
        per_chain.append(mcmc.get_samples())

    # stack chains and move samples to CPU to free GPU memory
    samples = {}
    for k in per_chain[0]:
        stacked = torch.stack([ch[k] for ch in per_chain])
        samples[k] = stacked.cpu() if (device is not None and device.type != "cpu") else stacked

    _empty_cache(device)
    return samples, info


def time_nuts(model, X, y, log_exposure, inits, n_iter=20, device=None, jit_compile=False):
    """Seconds per NUTS iteration (early warmup) on the specified device."""
    t0 = time.time()
    run_nuts(model, X, y, log_exposure, inits[:1], num_warmup=n_iter, num_samples=1,
             progress_bar=False, device=device, jit_compile=jit_compile)
    s = (time.time() - t0) / n_iter
    dev_str = str(device) if device else "current"
    print(f"{s:.2f} s / iteration on {dev_str}"
          f"  (early warmup iterations are not representative of tree depth)")
    return s


# ---------------------------------------------------------------- prediction
def _flat(samples):
    C, D = next(iter(samples.values())).shape[:2]
    return {k: v.reshape(C * D, *v.shape[2:]) for k, v in samples.items()}, C, D


@torch.no_grad()
def predict_from_samples(model, samples, X, log_exposure, batch=250, device=None):
    """For SMALL X (diagnostic subsets).

    Returns mu, rate, eta_glm, f_nn as [chains, draws, n].

    device : torch.device or None
        When given, model, X, log_exposure and sample chunks are moved to
        this device for the forward pass; results are returned on CPU.
    """
    if device is not None:
        model = to_device(device, model)
        X, log_exposure = to_device(device, X, log_exposure)

    flat, C, D = _flat(samples)
    sites = ("mu", "rate", "eta_glm")
    out = {s: [] for s in sites}
    for i in range(0, C * D, batch):
        chunk = {k: v[i:i + batch] for k, v in flat.items()}
        if device is not None:
            chunk = {k: v.to(device) for k, v in chunk.items()}

        pred = Predictive(model, posterior_samples=chunk, return_sites=sites)(
            X, log_exposure=log_exposure)
        for s in sites:
            out[s].append(pred[s].cpu() if device is not None else pred[s])

        _empty_cache(device)

    out = {s: torch.cat(v).reshape(C, D, -1) for s, v in out.items()}
    out["f_nn"] = torch.log(out["mu"]) - out["eta_glm"]
    return out


class _Running:
    """Streaming mean / sd of log(mu) and mean of mu."""
    def __init__(self):
        self.n = 0; self.s_mu = 0.; self.s = 0.; self.ss = 0.

    def add(self, mu):                                        # mu: [b, n]
        mu_cpu = mu.cpu()                                     # accumulate on CPU
        lm = torch.log(mu_cpu)
        self.n += mu_cpu.shape[0]
        self.s_mu = self.s_mu + mu_cpu.sum(0)
        self.s = self.s + lm.sum(0)
        self.ss = self.ss + (lm ** 2).sum(0)

    def result(self):
        m = self.s / self.n
        return {"mean_mu": self.s_mu / self.n, "mean_logmu": m,
                "sd_logmu": (self.ss / self.n - m ** 2).clamp(min=0).sqrt()}


@torch.no_grad()
def nuts_posterior_stats(model, samples, X, log_exposure, batch=250, device=None):
    if device is not None:
        model = to_device(device, model)
        X, log_exposure = to_device(device, X, log_exposure)

    flat, C, D = _flat(samples)
    acc = _Running()
    for i in range(0, C * D, batch):
        chunk = {k: v[i:i + batch] for k, v in flat.items()}
        if device is not None:
            chunk = {k: v.to(device) for k, v in chunk.items()}

        acc.add(Predictive(model, posterior_samples=chunk, return_sites=("mu",))(
            X, log_exposure=log_exposure)["mu"])

        _empty_cache(device)

    return acc.result()


@torch.no_grad()
def svi_posterior_stats(model, guide, X, log_exposure, num_samples=500, batch=100, device=None):
    if device is not None:
        model, guide = to_device(device, model, guide)
        X, log_exposure = to_device(device, X, log_exposure)

    acc = _Running()
    for i in range(0, num_samples, batch):
        b = min(batch, num_samples - i)
        acc.add(Predictive(model, guide=guide, num_samples=b, return_sites=("mu",))(
            X, log_exposure=log_exposure)["mu"])
        _empty_cache(device)
    return acc.result()


def deviance_from_stats(stats, y, log_exposure):
    """Deviance of the posterior-mean expected count."""
    return poisson_deviance(
        y_true=y.cpu(),
        y_pred=torch.log(stats["mean_mu"].cpu() * torch.exp(log_exposure.cpu())),
    ).item()


# ---------------------------------------------------------------- diagnostics
def _rhat_ess(x):   # x: [chains, draws, k]  — always on CPU by this point
    return (split_gelman_rubin(x, chain_dim=0, sample_dim=1),
            effective_sample_size(x, chain_dim=0, sample_dim=1))


@torch.no_grad()
def nuts_diagnostics(model, samples, info, X_eval, log_exposure_eval, y_eval=None,
                     n_points=200, seed=0, device=None):
    """R-hat / ESS on identifiable functions of the weights."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X_eval.shape[0], generator=g)[:n_points].to(X_eval.device)
    pred = predict_from_samples(model, samples, X_eval[idx], log_exposure_eval[idx], device=device)

    rows = {k: _rhat_ess(pred[k]) for k in ("mu", "eta_glm", "f_nn")}
    for n, v in samples.items():
        if "linear_pred" in n:
            rows[n] = _rhat_ess(v.reshape(v.shape[0], v.shape[1], -1))
    if y_eval is not None:
        ll = torch.distributions.Poisson(pred["rate"]).log_prob(y_eval[idx].cpu()).sum(-1, keepdim=True)
        rows["loglik (eval subset)"] = _rhat_ess(ll)

    dev_str = info.get("device", "unknown")
    print(f"Device used             : {dev_str}")
    print(f"Wall time per chain (s) : {[round(s, 1) for s in info['seconds']]}")
    print(f"Divergences per chain   : {info['divergences']}")
    print(f"Adapted step size       : {[round(s, 5) for s in info['step_size']]}")
    print(f"{'quantity':<40}{'max R-hat':>10}{'min ESS':>10}")
    for k, (rh, ess) in rows.items():
        print(f"{k:<40}{rh.max().item():>10.3f}{ess.min().item():>10.0f}")
    return rows


def width_by_bin(sd_nuts, sd_svi, feature, n_bins=10):
    """Mean posterior sd of log(mu) by quantile bin of one covariate."""
    feature = feature.cpu()
    sd_nuts, sd_svi = sd_nuts.cpu(), sd_svi.cpu()
    edges = torch.quantile(feature, torch.linspace(0, 1, n_bins + 1))
    b = (torch.bucketize(feature, edges[1:-1])).clamp(0, n_bins - 1)
    print(f"{'bin':>4}{'feature range':>24}{'n':>8}{'sd NUTS':>10}{'sd SVI':>10}{'ratio':>8}")
    for k in range(n_bins):
        m = b == k
        a, s = sd_nuts[m].mean().item(), sd_svi[m].mean().item()
        print(f"{k:>4}{f'[{edges[k]:.3g}, {edges[k+1]:.3g}]':>24}{int(m.sum()):>8}"
              f"{a:>10.4f}{s:>10.4f}{a/s:>8.2f}")