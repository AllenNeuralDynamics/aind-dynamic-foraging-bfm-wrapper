"""Split digests: one identity for the data a run trained and was scored on (wrapper #131).

Two runs share a digest only if they used the same data, whatever model family, loader path or
compute platform they came from. Each role (``train``, ``eval``, ``heldout_context``,
``heldout_eval``) gets two SHA-256 digests of canonical JSON:

- ``sessions_sha256``: which (subject id, original session id) pairs were used. A session counts
  as used if at least one of its trials is. Membership only, so a run that keeps ignored trials
  (``ignore_policy="include"``) and one that drops them match when they used the same sessions.
- ``trials_sha256``: the same, plus ``ignore_policy`` and each session's sorted trial ids. Strict
  identity.

The session key is the original id (``source_ses_idx`` when the multisubject builder rewrote
``ses_idx`` to ``"<subject>__<source>"``), so both loader paths give one digest. The role name is
not hashed: a digest names a set of data, so two runs that use it under different roles match.
Rows are sorted before hashing, so the order a loader lists sessions in does not matter.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from utils.multisubject import normalize_subject_id

SPLIT_DIGEST_VERSION = 1

logger = logging.getLogger(__name__)


def _sha256(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _trial_id(value: Any) -> Any:
    value = normalize_subject_id(value)
    return int(value) if isinstance(value, float) and value.is_integer() else value


def split_digests(
    raw: pd.DataFrame,
    roles: Mapping[str, Iterable[Any]],
    *,
    ignore_policy: str,
    partition: Mapping[str, tuple[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Digest each role's data.

    Parameters
    ----------
    raw : pd.DataFrame
        Trial rows with ``ses_idx``; ``subject_id``, ``source_ses_idx``, ``trial`` and
        ``animal_response`` are used when present (no ``trial`` column: row order in the session).
    roles : mapping
        Role name -> the session ids it uses, as they appear in ``raw["ses_idx"]``.
    ignore_policy : str
        ``"exclude"`` drops trials with ``animal_response == 2`` (no response), as the loaders do.
    partition : mapping, optional
        Role name -> ``(column, value)``: keep only that role's rows where ``raw[column] == value``
        (a split inside sessions, such as the external-bandit adapt prefix / test suffix).

    Returns
    -------
    dict
        Role -> ``{"sessions_sha256", "trials_sha256", "n_subjects", "n_sessions", "n_trials"}``.
    """
    known = set(raw["ses_idx"].astype(str))
    source = "source_ses_idx" if "source_ses_idx" in raw.columns else "ses_idx"
    out: dict[str, dict[str, Any]] = {}
    for role, ids in roles.items():
        wanted = {str(i) for i in ids}
        missing = sorted(wanted - known)
        if missing:
            raise ValueError(f"split digest: role {role!r} names sessions not in the data: {missing[:5]}")
        df = raw[raw["ses_idx"].astype(str).isin(wanted)]
        if partition and role in partition:
            column, value = partition[role]
            df = df[df[column] == value]
        if ignore_policy == "exclude" and "animal_response" in df.columns:
            df = df[df["animal_response"] != 2]
        subjects = df["subject_id"] if "subject_id" in df.columns else pd.Series("", index=df.index)
        sessions: dict[tuple[str, str], list[Any]] = {}
        for (subject, session), rows in df.groupby([subjects, df[source]], sort=False):
            key = (str(normalize_subject_id(subject)), str(normalize_subject_id(session)))
            trials = rows["trial"].map(_trial_id).tolist() if "trial" in rows.columns else list(range(len(rows)))
            sessions[key] = sorted(trials)
        keys = sorted(sessions)
        out[role] = {
            "sessions_sha256": _sha256({"v": SPLIT_DIGEST_VERSION, "sessions": [list(k) for k in keys]}),
            "trials_sha256": _sha256({"v": SPLIT_DIGEST_VERSION, "ignore_policy": ignore_policy,
                                      "sessions": [[*k, sessions[k]] for k in keys]}),
            "n_subjects": len({k[0] for k in keys}),
            "n_sessions": len(keys),
            "n_trials": sum(len(t) for t in sessions.values()),
        }
    return out


def bundle_split_digests(bundle: Any, *, heldout: bool = False) -> dict[str, dict[str, Any]]:
    """Digest a ``DatasetBundle``'s train / eval split, as its metadata names it.

    ``heldout=True`` names the roles ``heldout_context`` / ``heldout_eval`` (a held-out cohort's
    adapt and scored sessions). ``ignore_policy`` defaults to ``"exclude"`` when the metadata does
    not carry it, as the trainers read it.
    """
    meta = bundle.metadata
    names = ("heldout_context", "heldout_eval") if heldout else ("train", "eval")
    roles = {names[0]: meta.get("train_session_ids") or [], names[1]: meta.get("eval_session_ids") or []}
    partition = None
    column = meta.get("trial_partition_column")
    if column:
        partition = {names[0]: (column, meta.get("adapt_trial_partition", "adapt")),
                     names[1]: (column, meta.get("test_trial_partition", "test"))}
    return split_digests(bundle.raw, roles, ignore_policy=str(meta.get("ignore_policy", "exclude")),
                         partition=partition)


def log_split_digests(wandb_run: Any, digests: Mapping[str, Mapping[str, Any]]) -> None:
    """Write ``split/<role>/<key>`` summary values to a W&B run (nothing without a run)."""
    if wandb_run is None:
        return
    for role, values in digests.items():
        for key, value in values.items():
            wandb_run.summary[f"split/{role}/{key}"] = value


def record_split_digests(
    wandb_run: Any, compute: Callable[[], Mapping[str, Mapping[str, Any]]]
) -> Mapping[str, Mapping[str, Any]] | None:
    """Compute digests with ``compute()`` and log them. A failure is logged, never raised.

    The digest is provenance: it must not end a training run. A failure is visible as
    ``split/error`` on the run instead.
    """
    try:
        digests = compute()
    except Exception as exc:  # noqa: BLE001 - provenance must not stop training
        logger.warning("split digest not computed: %s", exc)
        if wandb_run is not None:
            wandb_run.summary["split/error"] = f"{type(exc).__name__}: {exc}"
        return None
    log_split_digests(wandb_run, digests)
    return digests
