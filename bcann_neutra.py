"""
NeuTra HMC (Hoffman et al., 2019) for the four BCANN models:
    PoissonBCANN, HorseshoePoissonBCANN   forward(X, y, log_exposure)
    GammaBCANN,   HorseshoeGammaBCANN     forward(X, y, weights)

NeuTra = "neural transport": fit a normalising flow q(theta) = T#N(0, I) to the posterior by
SVI, then run NUTS on z = T^{-1}(theta), whose posterior is close to N(0, I) if q is good.
The flow absorbs the correlations and scale differences (GLM block vs network weights, and
the horseshoe funnel w ~ N(0, tau * lambda)) that make plain NUTS take tiny steps and deep
trees. NUTS still targets the exact posterior: the flow is only a change of variables.

    1. guide, hist = neutra_fit_guide(model, Xs, ys, aux_s, glm_mle=glm_mle, num_steps=5000)
    2. samples, info = run_neutra(model, guide, Xs, ys, aux_s, num_chains=4)       # sequential
       samples, info = run_neutra_parallel(ModelCls, model_kwargs, guide, Xs, ys, aux_s,
                                           num_chains=4, devices=None)          # one process per chain
    3. neutra_diagnostics(model, samples, info, X_val, aux_val, y_val)

Flow VI alone (no NUTS), e.g. to compare it against NeuTra-NUTS on the same guide:
    guide, samples, hist = run_iaf_vi(model, Xs, ys, aux_s, glm_mle=glm_mle, num_steps=5000)
    stats = neutra_posterior_stats(model, samples, X_val, aux_val)

`aux` is log_exposure for the Poisson models and weights (claim counts) for the Gamma models,
i.e. whatever the 4th argument of run_nuts / gamma_run_nuts is.

`samples` has the same layout as run_nuts / gamma_run_nuts ({site: [chains, draws, ...]} on
CPU, in the model's constrained space), so predict_from_samples, nuts_posterior_stats,
gamma_nuts_posterior_stats, deviance_from_stats, width_by_bin, ... all take it unchanged.
info additionally holds "z": the base-space draws [chains, draws, latent_dim], whose R-hat/ESS
is the most direct check of the warped chain's mixing.

Guide: q is an affine map (loc, diagonal scale) composed with `num_transforms` inverse
autoregressive flows (IAFs). loc starts at the CANN start (GLM at the MLE, last network layer
at zero, Gamma shape at the Pearson estimate, horseshoe scales at their prior medians) and
every IAF starts at the identity, so before training q is the same Gaussian the other guides
start from. num_transforms=0 gives a diagonal Gaussian (NeuTra then only rescales the space).

Devices: build the model and guide under `with torch.device(device):` (the priors are created
at construction and model.to(device) does not move them), as in the NUTS examples.
run_neutra_parallel spreads chains round-robin over `devices` (default: every visible GPU),
so 4 chains on 4 GPUs run one per GPU, and 8 chains on 2 GPUs run 4 per GPU.
"""

import copy, math, os, re, time

import torch
import torch.multiprocessing as mp
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoContinuous
from pyro.infer.autoguide.initialization import init_to_value, init_to_sample
from pyro.infer.reparam import NeuTraReparam
from pyro.nn import PyroParam
from pyro.optim import ClippedAdam
from torch.distributions import biject_to
from pyro.distributions import constraints

from codebase_0 import set_seed
from gamma_bcann_vi import GammaBCANN, SHAPE_SITE
from gamma_horseshoe import HorseshoeGammaBCANN
from poisson_bcann_vi import _is_shrinkage, _kl_scaled
from poisson_bcann_hmc_nuts import (
    to_device, _empty_cache, _rhat_ess, nuts_diagnostics, nuts_posterior_stats,
    deviance_from_stats,
)
from gamma_bcann_hmc_nuts import (
    gamma_nuts_diagnostics, gamma_nuts_posterior_stats, gamma_deviance_from_stats,
)
from gamma_nuts_parallel import _last_log_line, _progress_printer


# ---------------------------------------------------------------- model dispatch
def _is_gamma(model):
    return isinstance(model, (GammaBCANN, HorseshoeGammaBCANN))


def _aux_kw(model, aux):
    """The model's 3rd forward argument: weights (Gamma) or log_exposure (Poisson)."""
    return {"weights": aux} if _is_gamma(model) else {"log_exposure": aux}


# ---------------------------------------------------------------- guide
class AutoNeuTraFlow(AutoContinuous):
    """
    q(theta) = T#N(0, I) on the model's unconstrained latent vector, with
        T = Affine(loc, scale) o IAF_K o ... o IAF_1

    loc starts at `init_loc_fn` (as in the other autoguides), scale at `init_scale`, and every
    IAF at the identity (its autoregressive network's output layer is zeroed, the flow
    analogue of the CANN zero-init of the last layer). AutoIAFNormal has no loc, so it ignores
    init_loc_fn and starts every chain of the flow far from the GLM.

    hidden_dim : list of IAF hidden layer widths; None -> [latent_dim].
    """

    def __init__(self, model, init_loc_fn=init_to_sample, init_scale=0.05,
                 num_transforms=2, hidden_dim=None):
        self._init_scale = init_scale
        self.num_transforms = num_transforms
        self.hidden_dim = hidden_dim
        super().__init__(model, init_loc_fn=init_loc_fn)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        self.loc = torch.nn.Parameter(self._init_loc())
        self.scale = PyroParam(self.loc.new_full((self.latent_dim,), self._init_scale),
                               constraint=constraints.softplus_positive)
        hidden = self.hidden_dim or [self.latent_dim]
        flows = []
        for _ in range(self.num_transforms):
            iaf = affine_autoregressive(self.latent_dim, hidden_dims=hidden)
            torch.nn.init.zeros_(iaf.arn.layers[-1].weight)
            torch.nn.init.zeros_(iaf.arn.layers[-1].bias)
            flows.append(iaf.to(self.loc.device))
        self.flows = torch.nn.ModuleList(flows)

    def get_base_dist(self):
        return dist.Normal(self.loc.new_zeros(()), 1.0).expand([self.latent_dim]).to_event(1)

    def get_transform(self, *args, **kwargs):
        return dist.transforms.ComposeTransform(
            # cache_size=1: log q(theta) in the ELBO inverts the transform, and the IAF's inverse
            # is a latent_dim-long sequential loop; with every part cached it reuses the
            # forward pass instead (the IAFs cache by default, AffineTransform does not)
            list(self.flows) + [dist.transforms.AffineTransform(self.loc, self.scale, event_dim=1,
                                                                cache_size=1)])


def neutra_init_values(model, X, y, aux, glm_mle=None, tau_init=None):
    """Constrained-space starting values: the CANN start used by every other guide.

    GLM block at glm_mle (if given), last network layer at zero, Gamma shape at glm_mle[2]
    (if given), horseshoe tau = tau_0 (or tau_init), c2 = slab_scale^2, lambda = 1.
    Everything else falls back to a prior draw.
    """
    with torch.no_grad():
        tr = poutine.trace(model).get_trace(X, y, **_aux_kw(model, aux))
    sites = {n: s["value"] for n, s in tr.nodes.items()
             if s["type"] == "sample" and not s["is_observed"]}
    values = {}
    if glm_mle is not None:
        for n in sites:
            if "linear_pred" in n:
                values[n] = glm_mle[0] if n.endswith("weight") else glm_mle[1]
        if len(glm_mle) > 2 and SHAPE_SITE in sites:
            values[SHAPE_SITE] = glm_mle[2].reshape(sites[SHAPE_SITE].shape)
    last = max(int(m.group(1)) for n in sites
               if (m := re.search(r"neural_net\.net\.(\d+)\.", n)))
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
    return {n: v.to(sites[n].device) for n, v in values.items()}


def build_neutra_guide(model, X, y, aux, glm_mle=None, num_transforms=2, hidden_dim=None,
                       init_scale=0.05, tau_init=None):
    """AutoNeuTraFlow for any of the four BCANNs, started at the CANN start.

    The constructor arguments are kept on guide.neutra_kwargs so run_neutra_parallel can
    rebuild the same guide in each process.
    """
    values = neutra_init_values(model, X, y, aux, glm_mle=glm_mle, tau_init=tau_init)
    guide = AutoNeuTraFlow(model, init_loc_fn=init_to_value(values=values, fallback=init_to_sample),
                           init_scale=init_scale, num_transforms=num_transforms,
                           hidden_dim=hidden_dim)
    guide(X, y, **_aux_kw(model, aux))      # run once so the prototype / parameters are created
    guide.neutra_kwargs = dict(num_transforms=num_transforms, hidden_dim=hidden_dim,
                               init_scale=init_scale)
    return guide


def neutra_fit_guide(model, X, y, aux, guide=None, glm_mle=None, num_steps=5000, lr=5e-3,
                     lr_final=None, flow_lr_mult=0.01, clip_norm=10.0, num_particles=1,
                     batch_size=None, report_every=250, max_restarts=5, random_state=None,
                     **guide_kwargs):
    """Fit the NeuTra flow by SVI on exactly the data NUTS will see (X, y, aux).

    batch_size : None -> full-batch ELBO. Otherwise minibatches with the prior and guide terms
        scaled by batch / n (kl_weight "uniform"), so each step is an unbiased full-ELBO estimate.
    lr_final : learning rate after num_steps (geometric decay); None -> constant lr.
    flow_lr_mult : the IAF networks train at lr * flow_lr_mult (loc and scale at lr). Adam moves
        every weight by ~lr per step, so an IAF output (a sum over ~latent_dim hidden units)
        moves by ~lr * latent_dim, i.e. several units per step at lr: the latent values blow
        up and the ELBO turns NaN within a few hundred steps (a few steps for the horseshoe).
    report_every : steps per report; the guide with the lowest mean -ELBO over a reporting
        window is restored at the end.
    max_restarts : on a non-finite loss (e.g. a horseshoe lambda drawn so small that its prior
        scale underflows to 0), go back to the guide at the last report, halve the learning
        rate (fresh Adam state) and carry on; give up after this many restarts.
    guide_kwargs : num_transforms, hidden_dim, init_scale, tau_init (see build_neutra_guide).

    Returns (guide, history) with history["neg_elbo_per_obs"] per reporting window and
    history["restarts"] the steps at which training was rolled back.
    """
    set_seed(random_state)
    if guide is None:
        guide = build_neutra_guide(model, X[:256], y[:256], None if aux is None else aux[:256],
                                   glm_mle=glm_mle, **guide_kwargs)
    n = X.shape[0]
    lrd = 1.0 if lr_final is None else (lr_final / lr) ** (1.0 / num_steps)

    def make_svi(lr_now):
        optim = ClippedAdam(lambda name: {"lr": lr_now * (flow_lr_mult if ".flows" in name else 1.0),
                                          "clip_norm": clip_norm, "lrd": lrd})
        return SVI(model, _kl_scaled(guide), optim, loss=Trace_ELBO(num_particles=num_particles))

    lr_now = lr
    svi = make_svi(lr_now)
    history = {"step": [], "neg_elbo_per_obs": [], "best_step": 0, "restarts": []}
    best, best_state, window = float("inf"), None, []
    good_state, good_step = copy.deepcopy(guide.state_dict()), 0
    t0 = time.time()
    for step in range(1, num_steps + 1):
        try:
            if batch_size is None:
                loss = svi.step(X, y, **_aux_kw(model, aux), kl_weight=1.0)
            else:
                idx = torch.randint(0, n, (batch_size,), device=X.device)
                # likelihood of the batch, rescaled to n rows, + full prior/guide terms -> * batch/n overall
                loss = svi.step(X[idx], y[idx], **_aux_kw(model, None if aux is None else aux[idx]),
                                kl_weight=batch_size / n) * n / batch_size
        except ValueError:                      # a distribution got NaN / invalid parameters
            loss = float("nan")
        if not math.isfinite(loss):
            if len(history["restarts"]) >= max_restarts:
                guide.load_state_dict(best_state if best_state is not None else good_state)
                raise RuntimeError(
                    f"non-finite ELBO at step {step} after {max_restarts} restarts "
                    f"(lr now {lr_now:.2e}); the guide has been reset to its best state. "
                    "Try a smaller lr or flow_lr_mult.")
            lr_now /= 2
            history["restarts"].append(step)
            print(f"step {step} | non-finite ELBO: back to step {good_step}, lr -> {lr_now:.2e}")
            guide.load_state_dict(good_state)
            svi, window = make_svi(lr_now), []
            continue
        window.append(loss)
        if step % report_every == 0 or step == num_steps:
            m = sum(window) / len(window) / n
            window = []
            history["step"].append(step)
            history["neg_elbo_per_obs"].append(m)
            good_state, good_step = copy.deepcopy(guide.state_dict()), step
            if m < best:
                best, best_state, history["best_step"] = m, good_state, step
            print(f"step {step} | -ELBO/n: {m:.4f} | {time.time() - t0:.0f}s")
    if best_state is not None:
        guide.load_state_dict(best_state)
    return guide, history


# ---------------------------------------------------------------- warped space <-> model sites
@torch.no_grad()
def neutra_transform(guide, z, batch=500):
    """Base-space draws z [..., latent_dim] -> {site: constrained values [..., *site_shape]}."""
    lead = z.shape[:-1]
    z = z.reshape(-1, z.shape[-1])
    T = guide.get_transform()
    out = {}
    for i in range(0, z.shape[0], batch):
        x = T(z[i:i + batch].to(guide.loc.device))
        for site, v in guide._unpack_latent(x):
            out.setdefault(site["name"], []).append(biject_to(site["fn"].support)(v).cpu())
    return {k: torch.cat(v).reshape(*lead, *v[0].shape[1:]) for k, v in out.items()}


def neutra_model(model, guide):
    """The model reparametrised by the flow: one latent site, the base-space vector z."""
    latent = {n for n, _ in guide.prototype_trace.iter_stochastic_nodes()}
    neutra = NeuTraReparam(guide)
    return poutine.reparam(model, config=lambda msg: neutra if msg["name"] in latent else None)


# ---------------------------------------------------------------- flow VI on its own (no NUTS)
@torch.no_grad()
def sample_flow(guide, num_draws=1000, num_chains=1, seed=0):
    """iid draws from the fitted flow q itself: z ~ N(0, I) -> T(z) -> constrained sites.

    Returns (samples, z) with samples in the run_neutra layout ({site: [chains, draws, ...]}
    on CPU), so neutra_posterior_stats / neutra_deviance_from_stats / width_by_bin take it
    unchanged. The draws are independent, so R-hat / ESS say nothing here.
    """
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(num_chains, num_draws, guide.latent_dim, generator=g, device="cpu")
    return neutra_transform(guide, z), z


def run_iaf_vi(model, X, y, aux, num_draws=1000, seed=0, **fit_kwargs):
    """IAF-flow VI as the posterior: neutra_fit_guide, then draws from q, no NUTS.

    fit_kwargs : everything neutra_fit_guide takes (glm_mle, num_steps, lr, batch_size,
        num_transforms, hidden_dim, random_state, ...).
    The returned guide is the same object run_neutra / run_neutra_parallel take, so NUTS
    can be run on it later without refitting.

    Returns (guide, samples, history), with history["z"] the base-space draws.
    """
    guide, history = neutra_fit_guide(model, X, y, aux, **fit_kwargs)
    samples, history["z"] = sample_flow(guide, num_draws=num_draws, seed=seed)
    return guide, samples, history


# ---------------------------------------------------------------- sampling
def run_neutra(model, guide, X, y, aux, num_chains=4, num_warmup=500, num_samples=500,
               target_accept=0.8, max_tree_depth=8, init_scale=1.0, seed=0,
               progress_bar=True, device=None, chain_ids=None, jit_compile=False):
    """NUTS in the flow's base space, one sequential chain after another.

    The warped posterior is ~N(0, I), so a diagonal mass matrix suffices (no full_mass block).
    init_scale : chain c starts at z0 = init_scale * N(0, I) (seed + c), i.e. a draw from q
        when 1.0; 0.0 starts every chain at the flow's median T(0).
    chain_ids : seeds / labels of the chains (default range(num_chains)); used by
        run_neutra_parallel so chain c in its own process gets seed + c.
    jit_compile : JIT-compile the potential energy (as in run_nuts / gamma_run_nuts).

    Returns (samples, info) laid out as run_nuts / gamma_run_nuts, with info["z"] the
    base-space draws [chains, draws, latent_dim]. Samples are returned on CPU.
    """
    if device is not None:
        model, guide = to_device(device, model, guide)
        X, y, aux = to_device(device, X, y, aux)
    chain_ids = list(range(num_chains)) if chain_ids is None else chain_ids
    kw = _aux_kw(model, aux)

    guide.requires_grad_(False)                   # the flow is fixed during sampling
    warped = neutra_model(model, guide)
    with torch.no_grad():
        z_name = next(n for n, s in poutine.trace(warped).get_trace(X, y, **kw).nodes.items()
                      if n.endswith("_shared_latent"))

    zs, info = [], {"divergences": [], "step_size": [], "seconds": [], "device": str(X.device)}
    try:
        for c in chain_ids:
            _empty_cache(device)
            g = torch.Generator().manual_seed(seed + c)
            z0 = (init_scale * torch.randn(guide.latent_dim, generator=g, device="cpu")).to(guide.loc)
            pyro.set_rng_seed(seed + c)
            kernel = NUTS(warped, target_accept_prob=target_accept, max_tree_depth=max_tree_depth,
                          init_strategy=init_to_value(values={z_name: z0}),
                          jit_compile=jit_compile, ignore_jit_warnings=True)
            mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=num_warmup,
                        num_chains=1, disable_progbar=not progress_bar)
            t0 = time.time()
            mcmc.run(X, y, **kw)
            info["seconds"].append(time.time() - t0)
            info["divergences"].append(len(mcmc.diagnostics()["divergences"]["chain 0"])
                                       if num_samples >= 4 else 0)
            info["step_size"].append(float(kernel.step_size))
            zs.append(mcmc.get_samples()[z_name].detach())
    finally:
        guide.requires_grad_(True)

    z = torch.stack(zs)                           # [chains, draws, latent_dim]
    samples = neutra_transform(guide, z)
    info["z"] = z.cpu()
    _empty_cache(device)
    return samples, info


def time_neutra(model, guide, X, y, aux, n_iter=20, device=None):
    """Seconds per NeuTra-NUTS iteration (early warmup) on the specified device."""
    t0 = time.time()
    run_neutra(model, guide, X, y, aux, num_chains=1, num_warmup=n_iter, num_samples=1,
               progress_bar=False, device=device)
    s = (time.time() - t0) / n_iter
    print(f"{s:.2f} s / iteration on {device or 'current'}"
          f"  (early warmup iterations are not representative of tree depth)")
    return s


# ---------------------------------------------------------------- one process per chain
def default_devices():
    """Every visible GPU, else the CPU."""
    n = torch.cuda.device_count()
    return [torch.device(f"cuda:{i}") for i in range(n)] if n else [torch.device("cpu")]


def _chain_worker(c, model_cls, model_kwargs, guide_kwargs, guide_state, X, y, aux, out_dir,
                  device, nuts_kwargs):
    # Send this process's stdout/stderr (incl. the progress bar) to its own log file;
    # output from spawned processes does not reach a notebook cell anyway.
    log = open(os.path.join(out_dir, f"chain_{c}.log"), "w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)

    torch.set_num_threads(1)              # one core per chain; chains don't fight for threads
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if device.type != "cpu":
        torch.set_default_device(device)  # build the priors on the device
    X, y = X.to(device), y.to(device)
    aux = None if aux is None else aux.to(device)
    model = model_cls(**model_kwargs)
    # Same architecture as the trained guide; the IAF permutations are buffers, so the
    # state dict restores them too.
    guide = build_neutra_guide(model, X[:256], y[:256], None if aux is None else aux[:256],
                               **guide_kwargs)
    guide.load_state_dict(guide_state)
    samples, info = run_neutra(model, guide, X, y, aux, chain_ids=[c], device=device,
                               **nuts_kwargs)
    torch.save({"samples": samples, "info": info}, os.path.join(out_dir, f"chain_{c}.pt"))


def load_neutra_chains(out_dir, num_chains):
    """Merge out_dir/chain_<c>.pt into run_neutra-style (samples, info)."""
    runs = [torch.load(os.path.join(out_dir, f"chain_{c}.pt")) for c in range(num_chains)]
    samples = {k: torch.cat([r["samples"][k] for r in runs]) for k in runs[0]["samples"]}
    info = {k: [v for r in runs for v in r["info"][k]]
            for k in ("divergences", "step_size", "seconds")}
    info["z"] = torch.cat([r["info"]["z"] for r in runs])
    info["device"] = [r["info"]["device"] for r in runs]
    return samples, info


def run_neutra_parallel(model_cls, model_kwargs, guide, X, y, aux, num_chains=4, devices=None,
                        out_dir="neutra_chains", seed=0, progress_every=10, **nuts_kwargs):
    """run_neutra with every chain in its own process, spread round-robin over `devices`.

    model_cls : PoissonBCANN, HorseshoePoissonBCANN, GammaBCANN or HorseshoeGammaBCANN
        (anything importable from a .py file with the same forward signature).
    model_kwargs : dict for model_cls(**model_kwargs) in each process. Use the same values
        as the model the guide was trained on.
    guide : trained guide from neutra_fit_guide / build_neutra_guide; its weights are sent to
        each process and loaded into a rebuilt copy there.
    devices : a device, or a list of devices (or strings); default every visible GPU, else
        CPU. Chain c runs on devices[c % len(devices)].
    seed : chain c uses seed + c, the same seeds run_neutra would use.
    progress_every : seconds between progress updates (one line per chain, read from its
        log). None turns it off.
    nuts_kwargs : num_warmup, num_samples, target_accept, max_tree_depth, init_scale,
        jit_compile.

    Each chain saves out_dir/chain_<c>.pt when it finishes; reload with
    load_neutra_chains(out_dir, num_chains).
    """
    os.makedirs(out_dir, exist_ok=True)
    if devices is None:
        devices = default_devices()
    elif isinstance(devices, (str, torch.device)):
        devices = [devices]                     # one device: every chain runs on it
    devices = [torch.device(d) for d in devices]

    # Hand CPU tensors to the children; each moves its copy to its device itself.
    # (CUDA tensors would go through CUDA IPC, which fails in many containers.)
    cpu = lambda t: None if t is None else t.detach().cpu()
    X, y, aux = cpu(X), cpu(y), cpu(aux)
    model_kwargs = {k: cpu(v) if torch.is_tensor(v) else v for k, v in model_kwargs.items()}
    guide_state = {k: cpu(v) for k, v in guide.state_dict().items()}

    ctx = mp.get_context("spawn")         # CUDA cannot be used in fork()ed processes
    procs = []
    t0 = time.time()
    for c in range(num_chains):
        dev = devices[c % len(devices)]
        p = ctx.Process(target=_chain_worker,
                        args=(c, model_cls, model_kwargs, guide.neutra_kwargs, guide_state,
                              X, y, aux, out_dir, dev, dict(nuts_kwargs, seed=seed)))
        p.start()
        procs.append(p)
    print(f"{num_chains} chains on {[str(devices[c % len(devices)]) for c in range(num_chains)]}")

    if progress_every:
        logs = [os.path.join(out_dir, f"chain_{c}.log") for c in range(num_chains)]
        show = _progress_printer()
        def report():
            m, s = divmod(int(time.time() - t0), 60)
            show("\n".join([f"elapsed {m // 60}:{m % 60:02d}:{s:02d}"] +
                           [f"chain {c}: {_last_log_line(log)}" for c, log in enumerate(logs)]))
        while any(p.is_alive() for p in procs):
            report()
            time.sleep(progress_every)
        report()
    for p in procs:
        p.join()

    failed = [c for c, p in enumerate(procs) if p.exitcode != 0]
    if failed:
        c = failed[0]
        try:
            with open(os.path.join(out_dir, f"chain_{c}.log"), errors="ignore") as f:
                tail = "".join(f.readlines()[-25:]).strip() or "<log is empty>"
        except FileNotFoundError:
            tail = "<no log: the process died before it started>"
        codes = {k: procs[k].exitcode for k in failed}
        raise RuntimeError(f"chains {failed} failed (exit codes {codes}; negative = killed by "
                           f"that signal, e.g. -9 = out of memory). chain {c} log:\n{tail}")

    samples, info = load_neutra_chains(out_dir, num_chains)
    info["wall_seconds"] = time.time() - t0
    return samples, info


# ---------------------------------------------------------------- diagnostics / summaries
@torch.no_grad()
def neutra_diagnostics(model, samples, info, X_eval, aux_eval, y_eval=None, n_points=200,
                       n_z=20, seed=0, device=None):
    """nuts_diagnostics / gamma_nuts_diagnostics, plus R-hat / ESS in the warped space.

    The z rows report the worst of n_z random base-space coordinates and of |z|^2, which
    is the quantity that mixes slowest when the flow misses the posterior's scale.
    """
    # The NUTS diagnostics draw their eval rows with a CPU generator, which fails when the
    # default device is CUDA (torch.set_default_device / `with torch.device(...)`). They move
    # the model, data and draws to `device` explicitly, so a CPU default here changes nothing else.
    with torch.device("cpu"):
        if _is_gamma(model):
            rows = gamma_nuts_diagnostics(model, samples, info, X_eval, aux_eval, y_eval,
                                          n_points=n_points, seed=seed, device=device)
        else:
            rows = nuts_diagnostics(model, samples, info, X_eval, aux_eval, y_eval,
                                    n_points=n_points, seed=seed, device=device)
    z = info["z"]
    g = torch.Generator().manual_seed(seed)
    k = torch.randperm(z.shape[-1], generator=g, device="cpu")[:n_z]
    for name, v in (("z (random coordinates)", z[..., k]),
                    ("|z|^2", (z ** 2).sum(-1, keepdim=True))):
        rows[name] = rh, ess = _rhat_ess(v)
        print(f"{name:<40}{rh.max().item():>10.3f}{ess.min().item():>10.0f}")
    return rows


def neutra_posterior_stats(model, samples, X, aux=None, batch=250, device=None):
    """nuts_posterior_stats (Poisson, uses log_exposure) or gamma_nuts_posterior_stats."""
    if _is_gamma(model):
        return gamma_nuts_posterior_stats(model, samples, X, batch=batch, device=device)
    return nuts_posterior_stats(model, samples, X, aux, batch=batch, device=device)


def neutra_deviance_from_stats(model, stats, y, aux):
    return (gamma_deviance_from_stats(stats, y, aux) if _is_gamma(model)
            else deviance_from_stats(stats, y, aux))
