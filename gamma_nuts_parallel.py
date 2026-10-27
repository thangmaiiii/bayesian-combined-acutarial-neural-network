"""
Gamma BCANN: HMC-NUTS with one process per chain.

Pyro's NUTS spends most of its time in Python, so a single chain keeps one CPU core busy
and leaves the GPU mostly idle. gamma_run_nuts runs its chains one after another; this
runs each chain in its own process (all sharing the GPU), so wall-clock time drops by
roughly the number of chains.

Usage (notebook or script):
    samples, info = gamma_run_nuts_parallel(
        dict(in_features=48, neural_net_params=nn_params),   # GammaBCANN(**model_kwargs)
        X_train, y_train, weights, inits, out_dir="nuts_chains",
        num_warmup=1000, num_samples=1000, target_accept=0.8, max_tree_depth=10,
        seed=seed, device=device)

For the horseshoe model pass model_cls=HorseshoeGammaBCANN (from gamma_horseshoe) and its
constructor arguments (tau_0, slab_scale, ...) in model_kwargs.

`samples` and `info` have the same layout as gamma_run_nuts's output. Each chain writes
its progress bar to out_dir/chain_<c>.log; while the chains run, the latest line of every
log is shown (redrawn in place in a notebook cell). Each chain saves its draws to
out_dir/chain_<c>.pt when it finishes, so finished chains survive a crash and can be
reloaded with gamma_load_chains(out_dir, num_chains).
"""

import os, re, time
import torch
import torch.multiprocessing as mp
from gamma_bcann_vi import GammaBCANN
from gamma_bcann_hmc_nuts import gamma_run_nuts


def _chain_worker(c, model_cls, model_kwargs, X, y, weights, init, out_dir, device, nuts_kwargs):
    # Send this process's stdout/stderr (incl. the progress bar) to its own log file;
    # output from spawned processes does not reach a notebook cell anyway.
    log = open(os.path.join(out_dir, f"chain_{c}.log"), "w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)

    torch.set_num_threads(1)              # one core per chain; chains don't fight for threads
    if device.type != "cpu":
        torch.set_default_device(device)  # build the priors on the device, as in the notebook
    model = model_cls(**model_kwargs)
    samples, info = gamma_run_nuts(model, X, y, weights, [init], device=device, **nuts_kwargs)
    torch.save({"samples": samples, "info": info}, os.path.join(out_dir, f"chain_{c}.pt"))


def _last_log_line(path):
    """Latest progress-bar state (or error message) in a chain's log."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            tail = f.read().decode(errors="ignore")
    except FileNotFoundError:
        return "starting"
    # tqdm redraws with \r, so the newest state is the last \r- or \n-separated piece
    parts = [s.strip() for s in re.split(r"[\r\n]", tail) if s.strip()]
    return parts[-1] if parts else "starting"


def _progress_printer():
    """Return show(text): redraws in place in a notebook cell, otherwise prints."""
    try:
        from IPython import get_ipython
        from IPython.display import clear_output
        if get_ipython() is not None:
            # clear_output + print rather than display(..., display_id=True).update():
            # VS Code can silently stop rendering display updates in a long-running cell.
            def show(text):
                clear_output(wait=True)
                print(text, flush=True)
            return show
    except ImportError:
        pass
    return lambda text: print(text + "\n", flush=True)


def gamma_load_chains(out_dir, num_chains):
    """Merge out_dir/chain_<c>.pt into gamma_run_nuts-style (samples, info)."""
    runs = [torch.load(os.path.join(out_dir, f"chain_{c}.pt")) for c in range(num_chains)]
    samples = {k: torch.cat([r["samples"][k] for r in runs]) for k in runs[0]["samples"]}
    info = {k: [v for r in runs for v in r["info"][k]]
            for k in ("divergences", "step_size", "seconds")}
    info["device"] = runs[0]["info"]["device"]
    return samples, info


def gamma_run_nuts_parallel(model_kwargs, X, y, weights, inits, out_dir="nuts_chains",
                            seed=0, device=None, model_cls=GammaBCANN, progress_every=10,
                            **nuts_kwargs):
    """gamma_run_nuts with every entry of `inits` run as its own process.

    model_kwargs : dict passed to model_cls(**model_kwargs) in each process. NUTS
        samples every model parameter, so only the constructor arguments need to be passed.
    model_cls : GammaBCANN, or any model with the same forward(X, y, weights) signature,
        e.g. gamma_horseshoe.HorseshoeGammaBCANN. Must be importable from a .py file.
    seed : chain c uses seed + c, the same seeds gamma_run_nuts would use.
    progress_every : seconds between progress updates (one line per chain, read from its
        log; redrawn in place in a notebook, printed as a block otherwise). None turns it off.
    nuts_kwargs : num_warmup, num_samples, target_accept, max_tree_depth, full_mass_glm.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device) if device is not None else X.device

    # Hand CPU tensors to the children; each moves its copy to `device` itself.
    cpu = lambda t: t.detach().cpu()
    X, y, weights = cpu(X), cpu(y), cpu(weights)
    # A CUDA tensor in model_kwargs (e.g. y_scale = gamma_weighted_mean(...) on the GPU)
    # would go to the children through CUDA IPC, which fails in many containers.
    model_kwargs = {k: cpu(v) if torch.is_tensor(v) else v for k, v in model_kwargs.items()}

    ctx = mp.get_context("spawn")         # CUDA cannot be used in fork()ed processes
    procs = []
    t0 = time.time()
    for c, init in enumerate(inits):
        kw = dict(nuts_kwargs, seed=seed + c)
        p = ctx.Process(target=_chain_worker,
                        args=(c, model_cls, model_kwargs, X, y, weights,
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

    samples, info = gamma_load_chains(out_dir, len(inits))
    info["wall_seconds"] = time.time() - t0
    return samples, info
