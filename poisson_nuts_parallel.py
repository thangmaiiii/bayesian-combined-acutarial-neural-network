"""
Poisson BCANN (horseshoe): HMC-NUTS with one process per chain.

Poisson counterpart of gamma_nuts_parallel.py: runs each chain of
poisson_bcann_hmc_nuts.run_nuts in its own process (all sharing the GPU), so wall-clock
time drops by roughly the number of chains. Every function takes `log_exposure` where
the Gamma version takes `weights`.

Usage (notebook or script):
    samples, info = run_nuts_parallel(
        dict(in_features=48, neural_net_params=nn_params, tau_0=TAU_0),  # HorseshoePoissonBCANN(**model_kwargs)
        X_train, y_train, log_exposure, inits, out_dir="nuts_chains_poisson_hs",
        num_warmup=1000, num_samples=1000, target_accept=0.9, max_tree_depth=10,
        seed=seed, device=device)

model_cls defaults to HorseshoePoissonBCANN; any model with forward(X, y, log_exposure)
works (e.g. poisson_bcann_vi.PoissonBCANN). From a .py script, put the call under
`if __name__ == "__main__":`, since spawned processes re-import the main script.

`samples` and `info` have the same layout as run_nuts's output. Progress, per-chain logs
(out_dir/chain_<c>.log) and saved chains (out_dir/chain_<c>.pt, reloadable with
load_chains(out_dir, num_chains)) work as in gamma_nuts_parallel.
"""

import os, time
import torch
import torch.multiprocessing as mp
from poisson_horseshoe import HorseshoePoissonBCANN
from poisson_bcann_hmc_nuts import run_nuts
from gamma_nuts_parallel import _last_log_line, _progress_printer
from gamma_nuts_parallel import gamma_load_chains as load_chains   # model-agnostic


def _chain_worker(c, model_cls, model_kwargs, X, y, log_exposure, init, out_dir, device,
                  nuts_kwargs):
    # Send this process's stdout/stderr (incl. the progress bar) to its own log file;
    # output from spawned processes does not reach a notebook cell anyway.
    log = open(os.path.join(out_dir, f"chain_{c}.log"), "w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)

    torch.set_num_threads(1)              # one core per chain; chains don't fight for threads
    if device.type != "cpu":
        torch.set_default_device(device)  # build the priors on the device
    model = model_cls(**model_kwargs)
    samples, info = run_nuts(model, X, y, log_exposure, [init], device=device, **nuts_kwargs)
    torch.save({"samples": samples, "info": info}, os.path.join(out_dir, f"chain_{c}.pt"))


def run_nuts_parallel(model_kwargs, X, y, log_exposure, inits, out_dir="nuts_chains_poisson",
                      seed=0, device=None, model_cls=HorseshoePoissonBCANN, progress_every=10,
                      **nuts_kwargs):
    """run_nuts with every entry of `inits` run as its own process.

    model_kwargs : dict passed to model_cls(**model_kwargs) in each process (in_features,
        neural_net_params, glm_prior_scale, tau_0, slab_scale, slab_df). Use the same
        values as the model the inits came from.
    model_cls : HorseshoePoissonBCANN, or any model with forward(X, y, log_exposure).
        Must be importable from a .py file.
    seed : chain c uses seed + c, the same seeds run_nuts would use.
    progress_every : seconds between progress updates (one line per chain, read from its
        log; redrawn in place in a notebook, printed as a block otherwise). None turns it off.
    nuts_kwargs : num_warmup, num_samples, target_accept, max_tree_depth, full_mass_glm.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device) if device is not None else X.device

    # Hand CPU tensors to the children; each moves its copy to `device` itself.
    cpu = lambda t: t.detach().cpu()
    X, y, log_exposure = cpu(X), cpu(y), cpu(log_exposure)
    # A CUDA tensor in model_kwargs would go to the children through CUDA IPC,
    # which fails in many containers.
    model_kwargs = {k: cpu(v) if torch.is_tensor(v) else v for k, v in model_kwargs.items()}

    ctx = mp.get_context("spawn")         # CUDA cannot be used in fork()ed processes
    procs = []
    t0 = time.time()
    for c, init in enumerate(inits):
        kw = dict(nuts_kwargs, seed=seed + c)
        p = ctx.Process(target=_chain_worker,
                        args=(c, model_cls, model_kwargs, X, y, log_exposure,
                              {n: cpu(v) for n, v in init.items()}, out_dir, device, kw))
        p.start()
        procs.append(p)

    if progress_every:
        logs = [os.path.join(out_dir, f"chain_{c}.log") for c in range(len(procs))]
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
        # Show the first failure here instead of only pointing at the logs. A missing or
        # empty log means the child died before _chain_worker ran (e.g. unpickling its
        # arguments), so its traceback went to the Jupyter server's terminal instead.
        c = failed[0]
        try:
            with open(os.path.join(out_dir, f"chain_{c}.log"), errors="ignore") as f:
                tail = "".join(f.readlines()[-25:]).strip() or "<log is empty>"
        except FileNotFoundError:
            tail = "<no log: the process died before it started>"
        codes = {k: procs[k].exitcode for k in failed}
        raise RuntimeError(f"chains {failed} failed (exit codes {codes}; negative = killed by "
                           f"that signal, e.g. -9 = out of memory). chain {c} log:\n{tail}")

    samples, info = load_chains(out_dir, len(inits))
    info["wall_seconds"] = time.time() - t0
    return samples, info
