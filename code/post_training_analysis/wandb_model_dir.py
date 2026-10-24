"""Materialise a runnable ``model_dir`` from a finished W&B run.

Every post-hoc analysis (`run_analysis generative`, `embedding`, held-out re-scoring, …)
needs the same thing: a directory holding ``inputs.yaml`` + ``outputs/checkpoints/`` that
``resolve_model_run`` can open. There are two ways to get one:

  * **Beaker result dataset** — mount the source task's dataset at ``/prior``. This is what
    ``launch_generative.py`` did. It identifies runs by enumerating Beaker *jobs*, which is a
    lossy proxy for "the grid": an experiment carries a replacement job for every preemption
    and every failed start, and a recovery re-submit lives in its own experiment. Both mistakes
    are silent.
  * **W&B run id** (this module) — the run id IS the provenance anchor, `group` + `state` is an
    exact description of the grid, and the artifact outlives the Beaker dataset.

This module is the second path, factored out of ``resume_heldout_beaker.py`` so any analysis can
use it. It preserves that tool's two hard-won details:

  1. ``inputs.yaml`` is NOT in the training-output artifact and must be reconstructed from the
     run config (``resolve_model_run`` needs the data+model blocks to rebuild the network and
     the reserved held-out set).
  2. ``checkpoints/index.json`` records each ``params_path`` as the ORIGINAL ABSOLUTE path.
     ``resolve_model_run`` remaps such a path by splitting on the FIRST ``/outputs/`` — and the
     original path can contain ``/outputs/`` twice — so the remap lands on a nonexistent file and
     **`best_eval` SILENTLY falls back to `final`**. Rewriting each entry to the relative form
     forces the resolver down its unambiguous ``startswith("outputs/")`` branch.

Fixed here vs. the original: the model type was **hardcoded to ``"gru"``**
(``mtype = "gru"  # ... disRNN would be "disrnn"``), so the W&B path could not restore a disRNN
run at all. It is now read from the run's own config.

CLI (inside a container that can reach W&B):

    python -m post_training_analysis.wandb_model_dir --run-id <rid> --dest /prior
    # -> /prior/run/{inputs.yaml, outputs/checkpoints/...}   (prints the model_dir)
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import urllib.request
from pathlib import Path

import wandb
import yaml

logger = logging.getLogger(__name__)

DEFAULT_ENTITY = "AIND-disRNN"


def fetch_run_config(entity: str, project: str, run_id: str) -> dict:
    """Fetch a run's config via the W&B GraphQL endpoint directly.

    Avoids ``wandb.Api().run()`` — that auto-loads the run's Sweep, and the Sweep.get() query is
    broken in the container's wandb build (raises 'Object of type Api is not JSON serializable').
    A plain GraphQL POST for the config field sidesteps the whole run/sweep object graph.
    """
    key = os.environ["WANDB_API_KEY"]
    q = ('query{project(name:"%s",entityName:"%s"){run(name:"%s"){config}}}'
         % (project, entity, run_id))
    body = json.dumps({"query": q}).encode()
    auth = base64.b64encode(f"api:{key}".encode()).decode()
    req = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Basic " + auth},
    )
    resp = json.load(urllib.request.urlopen(req))
    raw = resp["data"]["project"]["run"]["config"]
    return json.loads(raw) if isinstance(raw, str) else raw


def unwrap(d):
    """Strip W&B config's ``{"value":..., "desc":...}`` wrappers, recursively."""
    if isinstance(d, dict):
        if set(d.keys()) <= {"value", "desc"} and "value" in d:
            return unwrap(d["value"])
        return {k: unwrap(v) for k, v in d.items()}
    if isinstance(d, list):
        return [unwrap(x) for x in d]
    return d


def _write_inputs_yaml(cfg: dict, model_dir: Path) -> str:
    """Rebuild ``<model_dir>/inputs.yaml`` from the source run's config; return model type."""
    cfg = unwrap(cfg)
    for key in ("data", "model"):
        if key not in cfg or not isinstance(cfg[key], dict):
            raise ValueError(
                f"Source run config is missing a nested '{key}' block; cannot reconstruct "
                f"inputs.yaml (keys present: {sorted(cfg)[:20]})."
            )
    inputs = {"data": cfg["data"], "model": cfg["model"], "seed": cfg.get("seed")}
    (model_dir / "inputs.yaml").write_text(yaml.safe_dump(inputs, sort_keys=False))
    model_type = str(cfg["model"].get("type", "")).strip().lower()
    logger.info("Reconstructed inputs.yaml (model.type=%s, ignore_policy=%s, seed=%s)",
                model_type, cfg["data"].get("ignore_policy"), cfg.get("seed"))
    return model_type


def _normalize_index(outputs_dir: Path) -> None:
    """Rewrite index.json params_path entries to relative form.

    Without this, resolve_model_run's path remap can miss and `best_eval` silently degrades to
    `final` — a wrong-checkpoint bug that produces plausible numbers. See module docstring.
    """
    index_path = outputs_dir / "checkpoints" / "index.json"
    if not index_path.exists():
        logger.warning("No checkpoints/index.json; best_eval selection may fall back to final.")
        return
    index = json.loads(index_path.read_text())
    rewritten = 0
    for rec in index.get("checkpoints", []) or []:
        step = rec.get("step")
        if step is not None:
            rec["params_path"] = f"outputs/checkpoints/step_{step}/params.json"
            rewritten += 1
    index_path.write_text(json.dumps(index))
    logger.info("Normalized %d params_path entries in index.json to relative form.", rewritten)


def hydrate_model_dir(run_id: str, *, project: str, entity: str = DEFAULT_ENTITY,
                      dest: str | Path, model_type: str | None = None,
                      version: str = "latest") -> Path:
    """Download a finished run's checkpoints and assemble a runnable model_dir.

    Returns the model_dir (``<dest>/run``) — pass it straight to
    ``run_analysis ... --model-dir``.

    ``model_type`` is read from the run config unless given. It used to be hardcoded to "gru",
    which silently broke every non-GRU restore. ``version`` pins the artifact (``"v3"``) so a
    caller that already verified one version downloads that same version.
    """
    dest = Path(dest).expanduser()
    model_dir = dest / "run"
    outputs_dir = model_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # inputs.yaml first: it tells us the model type, which names the artifact.
    cfg = fetch_run_config(entity, project, run_id)
    cfg_model_type = _write_inputs_yaml(cfg, model_dir)
    model_type = (model_type or cfg_model_type).strip().lower()
    if model_type not in {"gru", "disrnn", "baseline_rl"}:
        raise ValueError(f"Unsupported/unknown model type {model_type!r} for run {run_id}.")

    artifact_ref = f"{entity}/{project}/{model_type}-output-{run_id}:{version}"
    logger.info("Downloading %s -> %s", artifact_ref, outputs_dir)
    wandb.Api().artifact(artifact_ref, type="training-output").download(root=str(outputs_dir))

    steps = sorted((outputs_dir / "checkpoints").glob("step_*"))
    if not steps:
        raise RuntimeError(f"No checkpoints/step_* under {outputs_dir} after download.")
    logger.info("Downloaded %d checkpoint(s); latest=%s", len(steps), steps[-1].name)

    _normalize_index(outputs_dir)
    return model_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True, help="Finished W&B run id.")
    p.add_argument("--project", required=True)
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--dest", required=True, help="Directory to assemble under (model_dir=<dest>/run).")
    p.add_argument("--model-type", default=None, help="Override; default reads it from the run config.")
    args = p.parse_args()
    model_dir = hydrate_model_dir(args.run_id, project=args.project, entity=args.entity,
                                  dest=args.dest, model_type=args.model_type)
    print(model_dir)


if __name__ == "__main__":
    main()
