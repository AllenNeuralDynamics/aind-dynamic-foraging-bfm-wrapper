from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Literal, Mapping, Optional

import numpy as np
import pandas as pd

import data_loaders.disrnn_dataset as dl
from disentangled_rnns.library import rnn_utils

from base.interfaces import DatasetLoader
from base.types import DatasetBundle
from utils.multisubject import (
    build_subject_index_maps,
    compute_train_eval_session_ids,
    merge_datasets_with_subject_index,
    normalize_subject_id,
    subject_sort_key,
    unique_subject_ids_preserve_order,
)
from utils.trial_timing_features import (
    attach_timing_features,
    create_disrnn_dataset_float,
    encode_timing_features,
    has_continuous_features,
    resolve_timing_config,
    shuffle_raw_response_columns,
    timing_feature_map,
)

# The disRNN library defaults when ``features`` is None: previous choice + previous
# reward. When we add timing features we must re-list these explicitly, because a
# non-empty ``features`` dict fully REPLACES the library default (it does not
# extend it) — so an omitted base feature would be silently dropped.
BASE_DISRNN_FEATURES: Mapping[str, str] = {
    "animal_response": "prev choice",
    "rewarded": "prev reward",
}

logger = logging.getLogger(__name__)


def _align_raw_df_with_valid_sessions(
    df: pd.DataFrame,
    *,
    ignore_policy: str,
) -> pd.DataFrame:
    """Drop sessions that would be silently removed by create_disrnn_dataset."""
    aligned = df.copy()
    if ignore_policy == "exclude" and "animal_response" in aligned.columns:
        valid_sessions = aligned.loc[aligned["animal_response"] != 2, "ses_idx"].unique()
        aligned = aligned[aligned["ses_idx"].isin(valid_sessions)].copy()
    return aligned


def _ordered_split_session_ids_from_aligned_df(
    df: pd.DataFrame,
    *,
    eval_every_n: int,
) -> tuple[list, list]:
    """Return train/eval session ids from aligned raw-session order."""
    if "ses_idx" not in df.columns:
        raise ValueError("Aligned raw dataframe must contain 'ses_idx'.")
    ordered_session_ids = list(dict.fromkeys(df["ses_idx"].tolist()))
    return compute_train_eval_session_ids(
        ordered_session_ids,
        eval_every_n=eval_every_n,
    )


def _normalize_snapshot_cols_to_retain(
    cols_to_retain: Optional[List[str]],
) -> list[str]:
    snapshot_cols_to_retain = list(cols_to_retain) if cols_to_retain is not None else None
    if snapshot_cols_to_retain is None:
        snapshot_cols_to_retain = [
            "trial",
            "subject_id",
            "ses_idx",
            "animal_response",
            "earned_reward",
            "curriculum_name",
        ]
    for required_column in ("trial", "subject_id", "ses_idx", "animal_response"):
        if required_column not in snapshot_cols_to_retain:
            snapshot_cols_to_retain.append(required_column)
    if "curriculum_name" not in snapshot_cols_to_retain:
        snapshot_cols_to_retain.append("curriculum_name")
    return snapshot_cols_to_retain


def _load_snapshot_selection(
    *,
    split: str = "train",
    subject_ids: Optional[List] = None,
    curricula: Optional[List[str]] = None,
    subject_ratio: Optional[object] = None,
    min_sessions: int = 10,
    heldout_every_n: int = 5,
    subject_sample_seed: Optional[int] = None,
    mature_only: bool = True,
    cols_to_retain: Optional[List[str]] = None,
    snapshot: Optional[str] = None,
) -> tuple[pd.DataFrame, list]:
    from utils.load_mice_database import load_mice_from_database  # noqa: PLC0415

    logger.info(
        "Loading %s dataset (mature_only=%s, snapshot=%s) …", split, mature_only, snapshot
    )
    return load_mice_from_database(
        split=split,
        subject_ids=subject_ids,
        curricula=curricula,
        subject_ratio=subject_ratio,
        min_sessions=min_sessions,
        heldout_every_n=heldout_every_n,
        seed=subject_sample_seed,
        mature_only=mature_only,
        cols_to_retain=_normalize_snapshot_cols_to_retain(cols_to_retain),
        snapshot=snapshot,
    )


def _resolve_multisubject_subject_order(
    df: pd.DataFrame,
    resolved_subject_ids: Iterable[int] | Iterable[str] | None = None,
) -> list:
    """Resolve a deterministic subject order for multisubject training."""
    available_subject_ids = unique_subject_ids_preserve_order(df["subject_id"].tolist())
    if resolved_subject_ids is None:
        return sorted(available_subject_ids, key=subject_sort_key)

    requested_subject_ids = unique_subject_ids_preserve_order(resolved_subject_ids)
    available_subject_ids_set = set(available_subject_ids)
    ordered_subject_ids = [
        normalize_subject_id(subject_id)
        for subject_id in requested_subject_ids
        if normalize_subject_id(subject_id) in available_subject_ids_set
    ]
    missing_subject_ids = [
        normalize_subject_id(subject_id)
        for subject_id in requested_subject_ids
        if normalize_subject_id(subject_id) not in available_subject_ids_set
    ]
    if missing_subject_ids:
        logger.warning(
            "Requested subject_ids contributed no trials to the fetched dataframe and will be "
            "skipped: %s. These subjects passed the >=min_sessions selection filter (which counts "
            "ALL sessions regardless of training stage) but yielded zero rows after the trial "
            "fetch. The usual cause is mature_only=True restricting the fetch to mature stages "
            "(STAGE_FINAL/GRADUATED): a subject with enough total sessions but no mature sessions "
            "fetches no trials. To keep such subjects, count mature sessions during selection or "
            "set mature_only=False.",
            missing_subject_ids,
        )
    return ordered_subject_ids


def _make_multisubject_session_ids_unique(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure session ids are unique across subjects in merged datasets."""
    if "subject_id" not in df.columns:
        raise ValueError(
            "Multisubject loading requires a 'subject_id' column in the raw dataframe."
        )
    if "ses_idx" not in df.columns:
        raise ValueError(
            "Multisubject loading requires a 'ses_idx' column in the raw dataframe."
        )

    normalized_df = df.copy()
    normalized_df["subject_id"] = normalized_df["subject_id"].map(normalize_subject_id)
    if "source_ses_idx" not in normalized_df.columns:
        normalized_df["source_ses_idx"] = normalized_df["ses_idx"]
    normalized_df["ses_idx"] = [
        f"{normalize_subject_id(subject_id)}__{normalize_subject_id(session_id)}"
        for subject_id, session_id in zip(
            normalized_df["subject_id"],
            normalized_df["source_ses_idx"],
        )
    ]
    return normalized_df


def _build_session_split_manifest_from_aligned_df(
    aligned_df: pd.DataFrame,
    *,
    eval_every_n: int,
) -> dict[str, object]:
    if aligned_df.empty:
        raise ValueError("Cannot build a session split manifest from an empty dataframe.")

    full_session_ids = list(dict.fromkeys(aligned_df["ses_idx"].tolist()))
    train_session_ids, eval_session_ids = _ordered_split_session_ids_from_aligned_df(
        aligned_df,
        eval_every_n=eval_every_n,
    )

    train_session_id_set = {str(session_id) for session_id in train_session_ids}
    eval_session_id_set = {str(session_id) for session_id in eval_session_ids}
    per_subject_rows = []
    if "subject_id" in aligned_df.columns:
        ordered_subject_ids = unique_subject_ids_preserve_order(aligned_df["subject_id"].tolist())
        for subject_id in ordered_subject_ids:
            subject_df = aligned_df[aligned_df["subject_id"] == subject_id]
            subject_session_ids = list(dict.fromkeys(subject_df["ses_idx"].tolist()))
            per_subject_rows.append(
                {
                    "subject_id": normalize_subject_id(subject_id),
                    "full_session_ids": [str(session_id) for session_id in subject_session_ids],
                    "train_session_ids": [
                        str(session_id)
                        for session_id in subject_session_ids
                        if str(session_id) in train_session_id_set
                    ],
                    "eval_session_ids": [
                        str(session_id)
                        for session_id in subject_session_ids
                        if str(session_id) in eval_session_id_set
                    ],
                }
            )

    return {
        "full_session_ids": [str(session_id) for session_id in full_session_ids],
        "train_session_ids": [str(session_id) for session_id in train_session_ids],
        "eval_session_ids": [str(session_id) for session_id in eval_session_ids],
        "per_subject": per_subject_rows,
    }


def _build_multisubject_session_split_manifest(
    df: pd.DataFrame,
    *,
    resolved_subject_ids: Iterable[int] | Iterable[str] | None,
    ignore_policy: str,
    eval_every_n: int,
) -> dict[str, object]:
    if "subject_id" not in df.columns:
        raise ValueError(
            "Multisubject loading requires a 'subject_id' column in the raw dataframe."
        )

    prepared_df = _make_multisubject_session_ids_unique(df)
    subject_order = _resolve_multisubject_subject_order(prepared_df, resolved_subject_ids)
    if not subject_order:
        raise ValueError("No subject data found for multisubject loading.")

    full_session_ids: list[str] = []
    train_session_ids: list[str] = []
    eval_session_ids: list[str] = []
    per_subject_rows: list[dict[str, object]] = []

    for subject_id in subject_order:
        subject_df = prepared_df[prepared_df["subject_id"] == subject_id].copy()
        if subject_df.empty:
            continue
        sort_columns = ["ses_idx"]
        if "trial" in subject_df.columns:
            sort_columns.append("trial")
        subject_df = subject_df.sort_values(sort_columns).reset_index(drop=True)
        subject_df = _align_raw_df_with_valid_sessions(
            subject_df,
            ignore_policy=ignore_policy,
        )
        if subject_df.empty:
            continue

        subject_session_ids = list(dict.fromkeys(subject_df["source_ses_idx"].tolist()))
        subject_train_ids, subject_eval_ids = compute_train_eval_session_ids(
            subject_session_ids,
            eval_every_n=eval_every_n,
        )
        full_session_ids.extend(str(session_id) for session_id in subject_session_ids)
        train_session_ids.extend(str(session_id) for session_id in subject_train_ids)
        eval_session_ids.extend(str(session_id) for session_id in subject_eval_ids)
        per_subject_rows.append(
            {
                "subject_id": normalize_subject_id(subject_id),
                "full_session_ids": [str(session_id) for session_id in subject_session_ids],
                "train_session_ids": [str(session_id) for session_id in subject_train_ids],
                "eval_session_ids": [str(session_id) for session_id in subject_eval_ids],
            }
        )

    if not full_session_ids:
        raise ValueError("No multisubject data remains after filtering valid sessions.")

    return {
        "full_session_ids": full_session_ids,
        "train_session_ids": train_session_ids,
        "eval_session_ids": eval_session_ids,
        "per_subject": per_subject_rows,
    }


def resolve_mice_snapshot_session_split_manifest(
    *,
    split: str = "train",
    subject_ids: Optional[List] = None,
    curricula: Optional[List[str]] = None,
    subject_ratio: Optional[object] = None,
    min_sessions: int = 10,
    heldout_every_n: int = 5,
    subject_sample_seed: Optional[int] = None,
    ignore_policy: str = "exclude",
    eval_every_n: int = 2,
    multisubject: bool = False,
    mature_only: bool = True,
    cols_to_retain: Optional[List[str]] = None,
    snapshot: Optional[str] = None,
) -> dict[str, object]:
    """Rebuild the exact train/eval session membership used during training."""

    df, selected_subject_ids = _load_snapshot_selection(
        split=split,
        subject_ids=subject_ids,
        curricula=curricula,
        subject_ratio=subject_ratio,
        min_sessions=min_sessions,
        heldout_every_n=heldout_every_n,
        subject_sample_seed=subject_sample_seed,
        mature_only=mature_only,
        cols_to_retain=cols_to_retain,
        snapshot=snapshot,
    )

    if multisubject:
        split_manifest = _build_multisubject_session_split_manifest(
            df,
            resolved_subject_ids=selected_subject_ids,
            ignore_policy=ignore_policy,
            eval_every_n=eval_every_n,
        )
    else:
        aligned_df = _align_raw_df_with_valid_sessions(df, ignore_policy=ignore_policy)
        if aligned_df.empty:
            raise ValueError("No snapshot sessions remain after applying ignore_policy.")
        split_manifest = _build_session_split_manifest_from_aligned_df(
            aligned_df,
            eval_every_n=eval_every_n,
        )

    return {
        "source": "snapshot_reconstruction",
        "split": split,
        "multisubject": bool(multisubject),
        "eval_every_n": int(eval_every_n),
        "snapshot": snapshot,
        "selected_subject_ids": [
            normalize_subject_id(subject_id) for subject_id in selected_subject_ids
        ],
        **split_manifest,
    }


def _build_multisubject_bundle(
    *,
    df: pd.DataFrame,
    resolved_subject_ids: Iterable[int] | Iterable[str] | None,
    ignore_policy: str,
    features: Mapping[str, str] | None,
    eval_every_n: int,
    batch_size: int | None,
    batch_mode: Literal["single", "rolling", "random"],
    metadata: Mapping[str, object],
    skip_subjects_with_insufficient_sessions: bool = False,
    session_split_by_subject: Mapping[str, Mapping[str, object]] | None = None,
    trial_split_by_subject: Mapping[str, Mapping[str, object]] | None = None,
    allow_unassigned_sessions: bool = False,
) -> DatasetBundle:
    """Build a merged multisubject dataset bundle from a trial dataframe.

    With ``allow_unassigned_sessions``, an explicit session split may leave sessions out of
    both lists: they stay in the full dataset (so every session keeps its session index)
    but are neither adaptation nor test sessions.
    """
    if session_split_by_subject is not None and trial_split_by_subject is not None:
        raise ValueError("Specify either a session split or a trial split, not both.")
    if "subject_id" not in df.columns:
        raise ValueError(
            "Multisubject loading requires a 'subject_id' column in the raw dataframe."
        )

    logger.info(
        "Building multisubject dataset from %d trials, %d raw subjects, columns=%s",
        len(df),
        int(df["subject_id"].nunique()) if "subject_id" in df.columns else -1,
        list(df.columns),
    )
    prepared_df = _make_multisubject_session_ids_unique(df)
    subject_order = _resolve_multisubject_subject_order(prepared_df, resolved_subject_ids)
    if not subject_order:
        raise ValueError("No subject data found for multisubject loading.")
    logger.info("Resolved multisubject training order: %s", subject_order)

    prepared_subjects: list[dict[str, object]] = []
    skipped_subjects_with_insufficient_sessions: list[dict[str, object]] = []
    # Partition once by subject. A per-subject ``== subject_id`` on the object-dtype
    # subject_id column re-scans every trial row in Python each iteration and
    # dominated load time (~250s for ~600 subjects). groupby(sort=False) preserves
    # within-group row order; we still iterate the explicit subject_order, so the
    # subject indexing/merge order is unchanged.
    subject_groups = {sid: g for sid, g in prepared_df.groupby("subject_id", sort=False)}
    for subject_id in subject_order:
        subject_df = subject_groups.get(subject_id)
        if subject_df is None or subject_df.empty:
            continue
        subject_df = subject_df.copy()
        sort_columns = ["ses_idx"]
        if "trial" in subject_df.columns:
            sort_columns.append("trial")
        subject_df = subject_df.sort_values(sort_columns).reset_index(drop=True)
        subject_df = _align_raw_df_with_valid_sessions(
            subject_df,
            ignore_policy=ignore_policy,
        )
        if subject_df.empty:
            logger.warning(
                "Skipping subject_id=%s because no valid sessions remain after filtering.",
                subject_id,
            )
            continue

        session_ids = list(dict.fromkeys(subject_df["ses_idx"].tolist()))
        source_session_ids = list(dict.fromkeys(subject_df["source_ses_idx"].tolist()))
        if len(source_session_ids) != len(session_ids):
            raise ValueError(
                "Expected source_ses_idx to preserve a 1:1 session mapping after alignment."
            )
        session_index_by_session_id = {
            str(session_id): int(index)
            for index, session_id in enumerate(session_ids, start=1)
        }
        trial_split_spec = None
        if session_split_by_subject is None and trial_split_by_subject is None:
            try:
                train_session_ids, eval_session_ids = compute_train_eval_session_ids(
                    session_ids,
                    eval_every_n=eval_every_n,
                )
            except ValueError as exc:
                if skip_subjects_with_insufficient_sessions:
                    logger.warning(
                        "Skipping subject_id=%s because only %d valid sessions remain after "
                        "filtering, which is insufficient for eval_every_n=%d: %s",
                        subject_id,
                        len(session_ids),
                        eval_every_n,
                        exc,
                    )
                    skipped_subjects_with_insufficient_sessions.append(
                        {
                            "subject_id": normalize_subject_id(subject_id),
                            "num_valid_sessions": int(len(session_ids)),
                            "eval_every_n": int(eval_every_n),
                            "reason": str(exc),
                        }
                    )
                    continue
                raise ValueError(
                    "Failed to construct a per-subject train/eval split for "
                    f"subject_id={subject_id!r}: only {len(session_ids)} valid sessions remain "
                    f"after filtering, with eval_every_n={eval_every_n}. "
                    "This commonly happens when low-ranked held-out subjects have too few mature "
                    "sessions. Choose subjects with more sessions, reduce eval_every_n in the "
                    "source training config, or specify explicit subject IDs."
                ) from exc
        elif session_split_by_subject is not None:
            subject_key = str(normalize_subject_id(subject_id))
            split_spec = session_split_by_subject.get(subject_key)
            if split_spec is None:
                raise ValueError(
                    "Explicit session split manifest has no entry for "
                    f"subject_id={subject_id!r}."
                )
            source_to_session_id = {
                str(normalize_subject_id(source_id)): session_id
                for source_id, session_id in zip(source_session_ids, session_ids)
            }
            if len(source_to_session_id) != len(source_session_ids):
                raise ValueError(
                    f"Subject {subject_id!r} has source session IDs that collide after "
                    "string normalization."
                )
            adapt_source_ids = [
                str(normalize_subject_id(value))
                for value in split_spec.get("adapt_session_ids", [])
            ]
            test_source_ids = [
                str(normalize_subject_id(value))
                for value in split_spec.get("test_session_ids", [])
            ]
            requested_ids = adapt_source_ids + test_source_ids
            if not adapt_source_ids or not test_source_ids:
                raise ValueError(
                    f"Subject {subject_id!r} must have at least one adaptation and one test "
                    "session in the explicit split manifest."
                )
            if len(set(requested_ids)) != len(requested_ids):
                raise ValueError(
                    f"Subject {subject_id!r} has duplicate or overlapping session IDs in the "
                    "explicit split manifest."
                )
            available_ids = set(source_to_session_id)
            unassigned_ok = allow_unassigned_sessions and set(requested_ids) <= available_ids
            if set(requested_ids) != available_ids and not unassigned_ok:
                missing = sorted(available_ids - set(requested_ids))
                unknown = sorted(set(requested_ids) - available_ids)
                raise ValueError(
                    f"Explicit session split for subject {subject_id!r} must cover every "
                    f"retained session exactly once; unassigned={missing}, unknown={unknown}."
                )
            train_session_ids = [source_to_session_id[value] for value in adapt_source_ids]
            eval_session_ids = [source_to_session_id[value] for value in test_source_ids]
        else:
            subject_key = str(normalize_subject_id(subject_id))
            trial_split_spec = trial_split_by_subject.get(subject_key)
            if trial_split_spec is None:
                raise ValueError(
                    "Explicit trial split manifest has no entry for "
                    f"subject_id={subject_id!r}."
                )
            requested_session_id = str(
                normalize_subject_id(trial_split_spec.get("session_id"))
            )
            normalized_source_ids = [
                str(normalize_subject_id(value)) for value in source_session_ids
            ]
            if normalized_source_ids != [requested_session_id]:
                raise ValueError(
                    f"Trial split for subject {subject_id!r} requires exactly session "
                    f"{requested_session_id!r}; retained sessions are {normalized_source_ids}."
                )
            total_trials = int(trial_split_spec.get("total_trials", -1))
            adapt_prefix_trials = int(
                trial_split_spec.get("adapt_prefix_trials", -1)
            )
            if total_trials != len(subject_df):
                raise ValueError(
                    f"Trial split for subject {subject_id!r} declares {total_trials} trials "
                    f"but the canonical table contains {len(subject_df)}."
                )
            if adapt_prefix_trials <= 0 or adapt_prefix_trials >= total_trials:
                raise ValueError(
                    f"Trial split for subject {subject_id!r} needs a non-empty prefix and "
                    "suffix."
                )
            train_session_ids = list(session_ids)
            eval_session_ids = list(session_ids)
            subject_df["external_split_partition"] = np.where(
                np.arange(len(subject_df)) < adapt_prefix_trials,
                "adapt",
                "test",
            )
        subject_df["subject_session_index"] = subject_df["ses_idx"].map(
            lambda session_id: session_index_by_session_id[str(session_id)]
        )
        subject_df["subject_max_session_index"] = int(len(session_ids))
        logger.info(
            "Prepared subject_id=%s with %d trials across %d sessions "
            "(train_sessions=%d, eval_sessions=%d).",
            subject_id,
            len(subject_df),
            len(session_ids),
            len(train_session_ids),
            len(eval_session_ids),
        )
        prepared_subjects.append(
            {
                "subject_id": subject_id,
                "df": subject_df,
                "full_session_ids": session_ids,
                "source_session_ids": source_session_ids,
                "train_session_ids": train_session_ids,
                "eval_session_ids": eval_session_ids,
                "session_index_by_session_id": session_index_by_session_id,
                "trial_split_spec": trial_split_spec,
            }
        )

    if not prepared_subjects:
        raise ValueError("No multisubject data remains after filtering valid sessions.")

    ordered_subject_ids, subject_id_to_index, index_to_subject_id = build_subject_index_maps(
        [item["subject_id"] for item in prepared_subjects]
    )
    logger.info("Subject-id to dense index map: %s", subject_id_to_index)

    subject_indices: list[int] = []
    full_datasets = []
    train_datasets = []
    eval_datasets = []
    ordered_frames: list[pd.DataFrame] = []
    full_session_ids: list = []
    train_session_ids: list = []
    eval_session_ids: list = []
    session_max_index_by_subject_index: list[int] = []
    session_context_rows: list[dict[str, object]] = []

    for item in prepared_subjects:
        subject_id = item["subject_id"]
        subject_index = subject_id_to_index[subject_id]
        subject_df = pd.DataFrame(item["df"]).copy()
        subject_df["subject_index"] = int(subject_index)
        session_index_by_session_id = {
            str(session_id): int(session_index)
            for session_id, session_index in dict(item["session_index_by_session_id"]).items()
        }
        session_max_index_by_subject_index.append(int(len(session_index_by_session_id)))
        session_context_rows.append(
            {
                "subject_id": normalize_subject_id(subject_id),
                "subject_index": int(subject_index),
                "ordered_session_ids": [str(session_id) for session_id in item["full_session_ids"]],
                "ordered_source_session_ids": [
                    str(session_id) for session_id in item["source_session_ids"]
                ],
            }
        )

        dataset = _create_disrnn_dataset(
            subject_df,
            ignore_policy=ignore_policy,
            features=features or None,
            batch_size=batch_size,
            batch_mode=batch_mode,
        )
        xs_subject = dataset.get_all()["xs"]
        expected_n_sessions = len(item["full_session_ids"])
        if int(xs_subject.shape[1]) != int(expected_n_sessions):
            raise ValueError(
                "Multisubject dataset construction changed the number of sessions for "
                f"subject {subject_id}. Expected {expected_n_sessions}, got {xs_subject.shape[1]}."
            )

        if session_split_by_subject is None and trial_split_by_subject is None:
            dataset_train, dataset_eval = rnn_utils.split_dataset(
                dataset,
                eval_every_n=eval_every_n,
            )
        elif session_split_by_subject is not None:
            session_index = {
                session_id: index
                for index, session_id in enumerate(item["full_session_ids"])
            }

            def _select_sessions(selected_session_ids):
                selected_indices = [session_index[value] for value in selected_session_ids]
                all_data = dataset.get_all()
                dataset_class = dataset.__class__
                return dataset_class(
                    all_data["xs"][:, selected_indices, :],
                    all_data["ys"][:, selected_indices, :],
                    y_type=dataset.y_type,
                    n_classes=dataset.n_classes,
                    x_names=dataset.x_names,
                    y_names=dataset.y_names,
                    batch_size=dataset.batch_size,
                    batch_mode=dataset.batch_mode,
                    rng=dataset.rng,
                )

            dataset_train = _select_sessions(item["train_session_ids"])
            dataset_eval = _select_sessions(item["eval_session_ids"])
        else:
            all_data = dataset.get_all()
            xs = np.asarray(all_data["xs"])
            ys = np.asarray(all_data["ys"])
            split_spec = item["trial_split_spec"]
            adapt_prefix_trials = int(split_spec["adapt_prefix_trials"])
            dataset_class = dataset.__class__
            dataset_train = dataset_class(
                xs[:adapt_prefix_trials, :, :],
                ys[:adapt_prefix_trials, :, :],
                y_type=dataset.y_type,
                n_classes=dataset.n_classes,
                x_names=dataset.x_names,
                y_names=dataset.y_names,
                batch_size=dataset.batch_size,
                batch_mode=dataset.batch_mode,
                rng=dataset.rng,
            )
            eval_ys = ys.copy()
            eval_ys[:adapt_prefix_trials, :, :] = -1
            dataset_eval = dataset_class(
                xs,
                eval_ys,
                y_type=dataset.y_type,
                n_classes=dataset.n_classes,
                x_names=dataset.x_names,
                y_names=dataset.y_names,
                batch_size=dataset.batch_size,
                batch_mode=dataset.batch_mode,
                rng=dataset.rng,
            )
        logger.info(
            "Built per-subject dataset for subject_id=%s (subject_index=%d): "
            "full=%s train=%s eval=%s",
            subject_id,
            subject_index,
            tuple(dataset._xs.shape),
            tuple(dataset_train._xs.shape),
            tuple(dataset_eval._xs.shape),
        )
        full_datasets.append(dataset)
        train_datasets.append(dataset_train)
        eval_datasets.append(dataset_eval)
        subject_indices.append(subject_index)
        ordered_frames.append(subject_df)
        full_session_ids.extend(item["full_session_ids"])
        train_session_ids.extend(item["train_session_ids"])
        eval_session_ids.extend(item["eval_session_ids"])

    merged_dataset = merge_datasets_with_subject_index(
        full_datasets,
        subject_indices,
        batch_size=batch_size,
        batch_mode=batch_mode,
    )
    merged_train_dataset = merge_datasets_with_subject_index(
        train_datasets,
        subject_indices,
        batch_size=batch_size,
        batch_mode=batch_mode,
    )
    merged_eval_dataset = merge_datasets_with_subject_index(
        eval_datasets,
        subject_indices,
        batch_size=batch_size,
        batch_mode=batch_mode,
    )

    raw_df = pd.concat(ordered_frames, ignore_index=True)
    raw_df["ses_idx"] = pd.Categorical(
        raw_df["ses_idx"],
        categories=full_session_ids,
        ordered=True,
    )
    sort_columns = ["ses_idx"]
    if "trial" in raw_df.columns:
        sort_columns.append("trial")
    raw_df = raw_df.sort_values(sort_columns).reset_index(drop=True)
    raw_df["ses_idx"] = raw_df["ses_idx"].astype(str)

    metadata_dict = dict(metadata)
    subject_curricula = {}
    if "curriculum_name" in raw_df.columns:
        for subject_id, subject_rows in raw_df.groupby("subject_id", sort=False):
            curricula = [
                normalize_subject_id(value)
                for value in subject_rows["curriculum_name"].dropna().unique().tolist()
            ]
            if not curricula:
                subject_curricula[normalize_subject_id(subject_id)] = "Unknown"
            elif len(curricula) == 1:
                subject_curricula[normalize_subject_id(subject_id)] = curricula[0]
            else:
                subject_curricula[normalize_subject_id(subject_id)] = "Mixed"
    metadata_dict.update(
        {
            "subject_ids": ordered_subject_ids,
            "subject_id_to_index": subject_id_to_index,
            "index_to_subject_id": index_to_subject_id,
            "num_subjects": int(len(ordered_subject_ids)),
            "session_max_index_by_subject_index": session_max_index_by_subject_index,
            "session_context": {
                "indexing": "1_based",
                "per_subject": session_context_rows,
            },
            "multisubject": True,
            "num_trials": int(len(raw_df)),
            "num_sessions": int(len(full_session_ids)),
            "full_session_ids": full_session_ids,
            "train_session_ids": train_session_ids,
            "eval_session_ids": eval_session_ids,
            "subject_curricula": subject_curricula,
            "batch_size": batch_size,
            "batch_mode": batch_mode,
            "skipped_subjects_with_insufficient_sessions": (
                skipped_subjects_with_insufficient_sessions
            ),
        }
    )
    logger.info(
        "Merged multisubject dataset with subject_index as feature 0: "
        "full=%s train=%s eval=%s x_names=%s",
        tuple(merged_dataset._xs.shape),
        tuple(merged_train_dataset._xs.shape),
        tuple(merged_eval_dataset._xs.shape),
        list(merged_dataset.x_names),
    )
    logger.info(
        "Raw multisubject dataframe columns=%s, subject_curricula=%s",
        list(raw_df.columns),
        subject_curricula,
    )

    return DatasetBundle(
        raw=raw_df,
        train_set=merged_train_dataset,
        eval_set=merged_eval_dataset,
        metadata=metadata_dict,
        extras={"dataset": merged_dataset},
    )


class MiceDatasetLoader(DatasetLoader):
    """Load the foraging dataset for mice experiments."""

    def __init__(
        self,
        subject_ids: Iterable[int],
        ignore_policy: str,
        features: Mapping[str, str],
        eval_every_n: int,
        multisubject: bool = False,
        batch_size: int | None = None,
        batch_mode: Literal["single", "rolling", "random"] = "random",
        seed: int | None = None,
        **extras: object,
    ) -> None:
        super().__init__(seed=seed)
        self.subject_ids = list(subject_ids)
        self.ignore_policy = ignore_policy
        self.features = dict(features)
        self.eval_every_n = eval_every_n
        self.multisubject = multisubject
        self.batch_size = batch_size
        self.batch_mode = batch_mode
        self.extras = extras

    def load(self) -> DatasetBundle:
        import aind_dynamic_foraging_data_utils.code_ocean_utils as co
        import aind_dynamic_foraging_multisession_analysis.multisession_load as ms_load

        # Fix numpy random seed (will affect batch_mode="random")
        if self.seed is not None:
            np.random.seed(self.seed)

        results = []
        for subject in self.subject_ids:
            logger.info("Querying docDB for {}".format(subject))
            subject_results = co.get_subject_assets(
                subject, modality=["behavior"], stage=["STAGE_FINAL", "GRADUATED"]
            )  # TODO, should we expose these filters?
            subject_results = subject_results.sort_values(
                by="session_name"
            ).reset_index(drop=True)
            results.append(subject_results)
        results = pd.concat(results)
        if len(results) == 0:
            raise Exception("No data found for subject ids")

        # TODO, filter sessions by performance

        logger.info("Getting s3 location")
        results = co.add_s3_location(results)

        logger.info("Loading NWBs")
        nwbs, df = ms_load.make_multisession_trials_df(results["s3_nwb_location"])

        if self.multisubject:
            metadata = {
                "ignore_policy": self.ignore_policy,
                "features": self.features,
                "eval_every_n": self.eval_every_n,
            }
            metadata.update(self.extras)
            return _build_multisubject_bundle(
                df=df,
                resolved_subject_ids=self.subject_ids,
                ignore_policy=self.ignore_policy,
                features=self.features,
                eval_every_n=self.eval_every_n,
                batch_size=self.batch_size,
                batch_mode=self.batch_mode,
                metadata=metadata,
            )

        dataset = dl.create_disrnn_dataset(
            df,
            ignore_policy=self.ignore_policy,
            features=self.features,
            batch_size=self.batch_size,
            batch_mode=self.batch_mode,
        )
        dataset_train, dataset_eval = rnn_utils.split_dataset(
            dataset, eval_every_n=self.eval_every_n
        )
        df = _align_raw_df_with_valid_sessions(df, ignore_policy=self.ignore_policy)
        train_session_ids, eval_session_ids = _ordered_split_session_ids_from_aligned_df(
            df,
            eval_every_n=self.eval_every_n,
        )
        metadata = {
            "subject_ids": self.subject_ids,
            "ignore_policy": self.ignore_policy,
            "features": self.features,
            "eval_every_n": self.eval_every_n,
            "num_trials": len(df),
            "num_sessions": int(df["ses_idx"].nunique()) if "ses_idx" in df else None,
            "train_session_ids": train_session_ids,
            "eval_session_ids": eval_session_ids,
            "multisubject": self.multisubject,
            "batch_size": self.batch_size,
            "batch_mode": self.batch_mode,
        }
        metadata.update(self.extras)

        extras = {"dataset": dataset}
        return DatasetBundle(
            raw=df,
            train_set=dataset_train,
            eval_set=dataset_eval,
            metadata=metadata,
            extras=extras,
        )


class MiceDatasetLoaderFromFile(DatasetLoader):
    """Load the foraging dataset from a pre-saved dataframe file (pkl or parquet)."""

    def __init__(
        self,
        file_path: str | Path,
        subject_ids: Iterable[int] | None = None,
        stages: Iterable[str] | None = None,
        ignore_policy: str = "none",
        features: Mapping[str, str] | None = None,
        eval_every_n: int = 10,
        multisubject: bool = False,
        batch_size: int | None = None,
        batch_mode: Literal["single", "rolling", "random"] = "random",
        seed: int | None = None,
        **extras: object,
    ) -> None:
        super().__init__(seed=seed)
        self.file_path = Path(file_path)
        self.subject_ids = list(subject_ids) if subject_ids is not None else None
        self.stages = list(stages) if stages is not None else None
        self.ignore_policy = ignore_policy
        self.features = dict(features) if features is not None else {}
        self.eval_every_n = eval_every_n
        self.multisubject = multisubject
        self.batch_size = batch_size
        self.batch_mode = batch_mode
        self.extras = extras

    def load(self) -> DatasetBundle:
        # Fix numpy random seed (will affect batch_mode="random")
        if self.seed is not None:
            np.random.seed(self.seed)

        # Load dataframe from file
        logger.info(f"Loading dataframe from {self.file_path}")
        if self.file_path.suffix == ".pkl":
            df = pd.read_pickle(self.file_path)
        elif self.file_path.suffix == ".parquet":
            df = pd.read_parquet(self.file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {self.file_path.suffix}. "
                "Supported formats are .pkl and .parquet"
            )
        
        logger.info(f"Loaded dataframe with {len(df)} trials")
        
        # Filter by subject_ids if provided
        if self.subject_ids is not None:
            if "subject_id" in df.columns:
                df = df[df["subject_id"].isin(self.subject_ids)]
                logger.info(
                    f"Filtered to {len(df)} trials for subject_ids: {self.subject_ids}"
                )
            else:
                logger.warning(
                    "subject_ids filter requested but 'subject_id' column not found in dataframe"
                )
        
        # Filter by stages if provided
        if self.stages is not None:
            if "stage" in df.columns:
                df = df[df["stage"].isin(self.stages)]
                logger.info(f"Filtered to {len(df)} trials for stages: {self.stages}")
            else:
                logger.warning(
                    "stages filter requested but 'stage' column not found in dataframe"
                )
        
        if len(df) == 0:
            raise Exception("No data found after filtering")

        if self.multisubject:
            metadata = {
                "file_path": str(self.file_path),
                "subject_ids": self.subject_ids,
                "stages": self.stages,
                "ignore_policy": self.ignore_policy,
                "features": self.features,
                "eval_every_n": self.eval_every_n,
            }
            metadata.update(self.extras)
            return _build_multisubject_bundle(
                df=df,
                resolved_subject_ids=self.subject_ids,
                ignore_policy=self.ignore_policy,
                features=self.features,
                eval_every_n=self.eval_every_n,
                batch_size=self.batch_size,
                batch_mode=self.batch_mode,
                metadata=metadata,
            )

        # Create dataset
        dataset = dl.create_disrnn_dataset(
            df,
            ignore_policy=self.ignore_policy,
            features=self.features,
            batch_size=self.batch_size,
            batch_mode=self.batch_mode,
        )
        dataset_train, dataset_eval = rnn_utils.split_dataset(
            dataset, eval_every_n=self.eval_every_n
        )
        df = _align_raw_df_with_valid_sessions(df, ignore_policy=self.ignore_policy)
        train_session_ids, eval_session_ids = _ordered_split_session_ids_from_aligned_df(
            df,
            eval_every_n=self.eval_every_n,
        )

        metadata = {
            "file_path": str(self.file_path),
            "subject_ids": self.subject_ids,
            "stages": self.stages,
            "ignore_policy": self.ignore_policy,
            "features": self.features,
            "eval_every_n": self.eval_every_n,
            "num_trials": len(df),
            "num_sessions": int(df["ses_idx"].nunique()) if "ses_idx" in df else None,
            "train_session_ids": train_session_ids,
            "eval_session_ids": eval_session_ids,
            "multisubject": self.multisubject,
            "batch_size": self.batch_size,
            "batch_mode": self.batch_mode,
        }
        metadata.update(self.extras)

        extras = {"dataset": dataset}
        return DatasetBundle(
            raw=df,
            train_set=dataset_train,
            eval_set=dataset_eval,
            metadata=metadata,
            extras=extras,
        )


def _create_disrnn_dataset(df, **kwargs):
    """Build a disRNN dataset, choosing a float-safe builder when needed.

    The upstream ``dl.create_disrnn_dataset`` allocates the input tensor as int64
    (``np.full(..., -1)``) and truncates float features. That is harmless for the
    integer-valued stock features but destroys continuous ones, so any feature set
    containing a continuous column is routed to the float-safe re-implementation.
    Integer-only feature sets keep calling upstream, so existing runs remain
    bit-for-bit reproducible. See utils.trial_timing_features for details.
    """
    if has_continuous_features(kwargs.get("features")):
        return create_disrnn_dataset_float(df, **kwargs)
    return dl.create_disrnn_dataset(df, **kwargs)


def _augment_features_with_timing(
    df: pd.DataFrame,
    *,
    base_features: Mapping[str, str] | None,
    run_seed: int | None = None,
    timing_features_cfg: object,
    snapshot: Optional[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Attach timing features to ``df`` and return the widened ``features`` map.

    When timing is disabled (default), returns ``df`` unchanged and a plain dict
    copy of ``base_features`` (possibly empty, meaning "use library defaults").

    When enabled, this:

    1. derives reaction-time / lick-count columns from the database (scoped to
       the subjects in ``df``) and merges them on ``(ses_idx, trial)``;
    2. encodes them (log RT; raw lick counts);
    3. returns a ``features`` map that EXPLICITLY lists the base disRNN inputs
       (prev choice + prev reward) plus the requested timing inputs — because a
       non-empty ``features`` dict replaces, rather than extends, the library
       default.
    """
    timing_cfg = resolve_timing_config(timing_features_cfg, run_seed=run_seed)
    features = dict(base_features) if base_features else {}
    if not timing_cfg.enabled:
        return df, features

    logger.info("Timing features enabled: %s", timing_cfg)
    df = attach_timing_features(
        df, snapshot=snapshot, lick_window_s=timing_cfg.lick_window_s
    )
    if timing_cfg.shuffle:
        # CONTROL ARM. Permute within session on the RAW columns, before encoding
        # and before the previous-trial shift, so the shuffled arm travels the
        # identical downstream path and differs from the real arm in exactly one
        # respect: trial alignment.
        df = shuffle_raw_response_columns(df, seed=timing_cfg.shuffle_seed)
    df = encode_timing_features(df, standardize=timing_cfg.standardize)

    # A non-empty features dict fully replaces the library default, so re-list the
    # base inputs explicitly, then add the timing inputs. Preserve any base
    # features the caller passed (e.g. a custom mapping) instead of the default.
    widened: dict[str, str] = dict(features) if features else dict(BASE_DISRNN_FEATURES)
    widened.update(timing_cfg.feature_map())
    logger.info("disRNN feature set widened to: %s", widened)
    return df, widened


class MiceSnapshotDatasetLoader(DatasetLoader):
    """Load mice behavioral data from the foraging database.

    Subjects are selected in one of two ways:

    * **Direct selection** – pass an explicit list via ``subject_ids`` (bypasses
      the selection pipeline entirely).
    * **Ratio-based pipeline** – subjects are filtered to those with at least
      ``min_sessions`` sessions, assigned their most-common ``task`` as a
      curriculum, ranked per curriculum by session count, a fixed ~20% heldout is
      reserved (every ``heldout_every_n``-th in rank order), and ``split="train"``
      then draws a seeded ``subject_ratio`` sample of the remaining pool.
      ``split="heldout"`` returns the reserved heldout set instead. See
      :func:`utils.load_mice_database.load_mice_from_database`.

    Train/eval splitting (within the selected subjects) is done by session order.
    In single-subject mode this follows ``rnn_utils.split_dataset`` directly; in
    multisubject mode the same every-``eval_every_n`` rule is applied independently
    within each subject before merging.

    Parameters
    ----------
    split:
        ``"train"`` (default) or ``"heldout"`` — which selection partition to load.
    subject_ids:
        Explicit list of subject IDs to use. When provided, the selection pipeline
        is bypassed.
    curricula:
        Chosen ``task`` names to include. Defaults to the three standard foraging
        curricula (Coupled Baiting, Uncoupled Baiting, Uncoupled Without Baiting).
    subject_ratio:
        Per-curriculum fraction (``{task: float}``; scalar broadcasts) of the
        post-heldout pool to sample for training. ``None`` -> 1.0 each.
    min_sessions:
        Minimum (all-stage) session count for a subject to be eligible.
    heldout_every_n:
        Reserve every ``heldout_every_n``-th ranked subject per curriculum as
        heldout (default 5 -> ~20%).
    subject_sample_seed:
        Seed for the training-set subject sample (heldout is seed-independent).
    ignore_policy:
        Ignore-policy string passed to :func:`dl.create_disrnn_dataset`.
    features:
        Feature mapping passed to :func:`dl.create_disrnn_dataset`.
    eval_every_n:
        Every ``eval_every_n``-th trial goes to the eval split.
    multisubject:
        If ``True``, build one dataset per subject, split sessions within each
        subject, and merge them with ``subject_index`` prepended as feature 0.
    mature_only:
        If ``True`` *(default)*, restrict the returned trials to sessions in stage
        ``STAGE_FINAL`` or ``GRADUATED``. Does NOT affect subject selection.
    cols_to_retain:
        Trial-level columns to keep. Falls back to the module default when
        ``None``.
    snapshot:
        Pin reads to a dated database snapshot (e.g. ``"20260603"``) instead of
        the live database, for reproducibility. ``None`` (default) reads live.
    batch_size:
        Batch size for the disRNN dataset iterator.
    batch_mode:
        One of ``"single"``, ``"rolling"``, or ``"random"``.
    seed:
        Random seed forwarded to the base class and NumPy (affects batch sampling).
    """

    def __init__(
        self,
        split: str = "train",
        subject_ids: Optional[List] = None,
        curricula: Optional[List[str]] = None,
        subject_ratio: Optional[object] = None,
        min_sessions: int = 10,
        heldout_every_n: int = 5,
        subject_sample_seed: Optional[int] = None,
        ignore_policy: str = "exclude",
        features: Optional[Mapping[str, str]] = None,
        eval_every_n: int = 2,
        multisubject: bool = False,
        mature_only: bool = True,
        cols_to_retain: Optional[List[str]] = None,
        snapshot: Optional[str] = None,
        batch_size: Optional[int] = None,
        batch_mode: Literal["single", "rolling", "random"] = "random",
        seed: Optional[int] = None,
        **extras: object,
    ) -> None:
        super().__init__(seed=seed)
        self.split = split
        self.subject_ids = list(subject_ids) if subject_ids is not None else None
        self.curricula = curricula
        self.subject_ratio = subject_ratio
        self.min_sessions = min_sessions
        self.heldout_every_n = heldout_every_n
        self.subject_sample_seed = subject_sample_seed
        self.mature_only = mature_only
        self.cols_to_retain = cols_to_retain
        self.snapshot = snapshot
        self.ignore_policy = ignore_policy
        self.features = dict(features) if features is not None else {}
        self.eval_every_n = eval_every_n
        self.multisubject = multisubject
        self.batch_size = batch_size
        self.batch_mode = batch_mode
        self.extras = extras

    def load(self) -> DatasetBundle:
        if self.seed is not None:
            np.random.seed(self.seed)

        df, subject_ids = _load_snapshot_selection(
            split=self.split,
            subject_ids=self.subject_ids,
            curricula=self.curricula,
            subject_ratio=self.subject_ratio,
            min_sessions=self.min_sessions,
            heldout_every_n=self.heldout_every_n,
            subject_sample_seed=self.subject_sample_seed,
            mature_only=self.mature_only,
            cols_to_retain=self.cols_to_retain,
            snapshot=self.snapshot,
        )

        # Optionally derive + attach reaction-time / lick-count inputs and widen
        # the disRNN feature set. No-op unless data.timing_features.enabled.
        df, self.features = _augment_features_with_timing(
            df,
            base_features=self.features,
            timing_features_cfg=self.extras.get("timing_features"),
            snapshot=self.snapshot,
            # Ties the shuffled-arm permutation to the run seed, so seed
            # replicates of that arm see DIFFERENT permutations.
            run_seed=self.seed,
        )

        if self.multisubject:
            metadata = {
                "subject_ids": subject_ids,
                "split": self.split,
                "subject_ratio": self.subject_ratio,
                "min_sessions": self.min_sessions,
                "heldout_every_n": self.heldout_every_n,
                "subject_sample_seed": self.subject_sample_seed,
                "mature_only": self.mature_only,
                "curricula": self.curricula,
                "ignore_policy": self.ignore_policy,
                "features": self.features,
                "snapshot": self.snapshot,
            }
            metadata.update(self.extras)
            return _build_multisubject_bundle(
                df=df,
                resolved_subject_ids=subject_ids,
                ignore_policy=self.ignore_policy,
                features=self.features,
                eval_every_n=self.eval_every_n,
                batch_size=self.batch_size,
                batch_mode=self.batch_mode,
                metadata=metadata,
            )

        logger.info("Building disRNN datasets …")
        dataset = _create_disrnn_dataset(
            df,
            ignore_policy=self.ignore_policy,
            features=self.features or None,  # pass None to use library defaults when empty
            batch_size=self.batch_size,
            batch_mode=self.batch_mode,
        )
        dataset_train, dataset_eval = rnn_utils.split_dataset(
            dataset, eval_every_n=self.eval_every_n
        )
        df = _align_raw_df_with_valid_sessions(df, ignore_policy=self.ignore_policy)
        split_manifest = _build_session_split_manifest_from_aligned_df(
            df,
            eval_every_n=int(self.eval_every_n),
        )

        metadata = {
            "subject_ids": subject_ids,
            "split": self.split,
            "subject_ratio": self.subject_ratio,
            "min_sessions": self.min_sessions,
            "heldout_every_n": self.heldout_every_n,
            "subject_sample_seed": self.subject_sample_seed,
            "mature_only": self.mature_only,
            "curricula": self.curricula,
            "ignore_policy": self.ignore_policy,
            "features": self.features,
            "snapshot": self.snapshot,
            "num_trials": len(df),
            "num_sessions": (
                int(df["ses_idx"].nunique()) if "ses_idx" in df.columns else None
            ),
            "train_session_ids": split_manifest["train_session_ids"],
            "eval_session_ids": split_manifest["eval_session_ids"],
            "multisubject": self.multisubject,
            "batch_size": self.batch_size,
            "batch_mode": self.batch_mode,
        }
        metadata.update(self.extras)

        return DatasetBundle(
            raw=df,
            train_set=dataset_train,
            eval_set=dataset_eval,
            metadata=metadata,
            extras={"dataset": dataset},
        )
