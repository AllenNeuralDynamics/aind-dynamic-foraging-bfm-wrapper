"""Figures for one hierarchical-Bayes fit: sampler trustworthiness, then the science.

Split by audience, because the two differ. The diagnostic set decides whether the numbers
may be quoted at all; the reading set is what goes in a talk. Both are logged from inside
the run, so a fit that cost hours never has to be repeated to see it.

**No arviz API is used here, deliberately.** `arviz` is an unbounded dependency of the
models `[bayes]` extra, and its 0.x and 1.x lines disagree on the basics: `InferenceData`
is a class on 0.x and an alias for `xarray.DataTree` on 1.x, `.groups` is a method
returning bare names on 0.x and a property returning node paths on 1.x. Figure code that
touches those breaks on whichever line the image did not happen to resolve -- which is
exactly how `save_fit` came to crash every HB run (models #64/#68). Everything below reads
plain `xarray.Dataset` objects and draws with matplotlib, so there is no arviz version to
be wrong about.

What is deliberately NOT here, and why:

``plot_shrinkage``
    Needs an *unpooled* per-subject arm to make its point -- its own docstring says so:
    without it the figure shows where subjects ended up, not that pooling moved them. The
    HB run produces no unpooled estimates, so a faithful version needs a per-subject MLE
    pass (``baseline_rl_hattori``) alongside. Omitted rather than shipped degenerate.

Per-session latent trajectories
    Need the session-level sites, which ``save_fit`` excludes by default, plus a replay of
    the Q recursion. Tracked separately; the likelihood is not reshaped to produce a figure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Order matches HATTORI2019_PARAMS; the last is unbounded and takes no transform.
PARAM_NAMES = (
    "learn_rate_rew",
    "learn_rate_unrew",
    "forget_rate_unchosen",
    "softmax_inverse_temperature",
    "bias_l",
)

#: Fixed x-axis range per parameter, in the model's own (bounded) units.
#:
#: Mirrors ``hierarchical_bayes.plotting.PARAM_XLIM``, which is the source of truth. The
#: duplication is deliberate and matches how this module already treats ``PARAM_NAMES`` and
#: ``to_bounded``: it exists to log figures without importing the models plotting stack, so
#: importing that constant would defeat its purpose and couple in-run logging to a models
#: version. Keep the two in step when either changes.
#:
#: Why fixed at all: autoscaling to the posterior's own mass makes every panel look equally
#: well determined and makes rungs incomparable -- a parameter pinned at 0.02 and one spread
#: across half its support render at the same width, and the same parameter changes scale
#: between D=10 and D=614. Fixing the axis puts the width of the posterior on the page.
#:
#: The three rates are probabilities on (0, 1) and the inverse temperature is bounded by
#: ``beta_max``, so those are supports rather than choices, and the temperature panel doubles
#: as a check on whether the posterior is pressing against that ceiling. ``bias_l`` is
#: unbounded; +/-0.5 is a reporting convention, not a limit.
PARAM_XLIM = {
    "learn_rate_rew": (0.0, 1.0),
    "learn_rate_unrew": (0.0, 1.0),
    "forget_rate_unchosen": (0.0, 1.0),
    "softmax_inverse_temperature": (0.0, None),   # None -> beta_max
    "bias_l": (-0.5, 0.5),
}


def _param_xlim(name, beta_max=10.0):
    """Canonical ``(lo, hi)`` for a parameter, or ``None`` if it has no convention."""
    span = PARAM_XLIM.get(name)
    if span is None:
        return None
    lo, hi = span
    return (lo, float(beta_max) if hi is None else hi)


def load_fit(netcdf_path, sample_stats_path=None):
    """Read a saved fit back as ``{group_name: xarray.Dataset}``.

    Handles both on-disk layouts, since fits written before models #64 stored each group at
    the netCDF root rather than in a named group:

    * grouped (current) -- ``/posterior`` and ``/sample_stats``
    * flat (legacy) -- posterior variables at the root, diagnostics in a sidecar file

    Returns
    -------
    dict of str to xarray.Dataset
        Always contains ``"posterior"``; contains ``"sample_stats"`` when diagnostics were
        found. Never an arviz object -- see the module docstring.
    """
    import xarray as xr

    def _read(path, group=None):
        """Open, materialise, and close -- so no file handle outlives this call.

        `open_dataset` returns a lazily-backed Dataset holding the netCDF file open. The
        plotting code materialises every array it touches immediately, so there is nothing
        to gain from laziness and a descriptor to lose: a long-lived process regenerating
        figures would accumulate open handles until it hit its limit.
        """
        with xr.open_dataset(str(path), group=group) as ds:
            return ds.load()

    netcdf_path = str(netcdf_path)
    groups = {}
    try:
        groups["posterior"] = _read(netcdf_path, group="posterior")
        try:
            groups["sample_stats"] = _read(netcdf_path, group="sample_stats")
        except (OSError, KeyError):
            pass
    except (OSError, KeyError):
        logger.info("Fit at %s is in the pre-grouped layout; reading the root.", netcdf_path)
        groups["posterior"] = _read(netcdf_path)

    if "sample_stats" not in groups and sample_stats_path and Path(sample_stats_path).exists():
        groups["sample_stats"] = _read(sample_stats_path)
    return groups


def to_bounded(unconstrained, param_index, beta_max=10.0):
    """Map one parameter from the sampler's scale to the one it is named for."""
    from scipy.stats import norm

    values = np.asarray(unconstrained)
    if param_index == 4:            # side bias is unbounded
        return values
    if param_index == 3:            # inverse temperature
        return norm.cdf(values) * beta_max
    return norm.cdf(values)         # the three rates, on [0, 1]


def _draws(posterior, site="population_mean"):
    """(chain, draw, param) array for one site, with the chain axis kept."""
    values = np.asarray(posterior[site])
    if values.ndim == 2:            # (draw, param) -- a single unlabelled chain
        values = values[None, ...]
    return values


def plot_sampler_diagnostics(groups, path=None):
    """Traces, between-chain rank uniformity, and the energy distribution.

    These decide whether a fit's numbers may be used: a trace that has not mixed, ranks
    that are not uniform across chains, or a narrow energy distribution against wide
    transitions all mean the posterior being reported is not the posterior.

    Drawn from the raw arrays rather than through ``az.plot_rank`` / ``az.plot_energy``,
    for the version-independence reason in the module docstring.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    posterior = groups["posterior"]
    stats = groups.get("sample_stats")
    draws = _draws(posterior)
    n_chains, n_draws, n_params = draws.shape
    has_energy = stats is not None and "energy" in stats

    panels = ["trace"] + (["rank"] if n_chains > 1 else []) + (["energy"] if has_energy else [])
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, 2.6 * len(panels)), squeeze=False)

    for ax, kind in zip(axes[:, 0], panels):
        if kind == "trace":
            for chain in range(n_chains):
                for p in range(n_params):
                    ax.plot(draws[chain, :, p], lw=0.7, alpha=0.8)
            ax.set_title(
                f"population_mean traces · {n_chains} chain{'s' if n_chains > 1 else ''}"
                f" × {n_draws} draws",
                loc="left", fontsize=9,
            )
            ax.set_xlabel("draw")
        elif kind == "rank":
            # Pool every draw of a parameter, rank them, and histogram each chain's ranks.
            # Well-mixed chains each cover the pooled range uniformly; a chain stuck in its
            # own region shows up as a skewed block.
            #
            # The ranked parameter and the title come from the same index, so the panel
            # cannot end up labelled with a parameter it did not rank if PARAM_NAMES is
            # reordered or the site's width changes.
            flat = draws.reshape(n_chains * n_draws, n_params)
            ranked = 0
            ranked_name = (
                PARAM_NAMES[ranked] if ranked < len(PARAM_NAMES) else f"param {ranked}"
            )
            order = np.argsort(np.argsort(flat[:, ranked]))
            ranks = order.reshape(n_chains, n_draws)
            bins = np.linspace(0, n_chains * n_draws, min(20, n_draws) + 1)
            for chain in range(n_chains):
                ax.hist(ranks[chain], bins=bins, histtype="step", lw=1.2,
                        label=f"chain {chain}")
            ax.axhline(n_draws / (len(bins) - 1), color="0.4", ls="--", lw=0.9,
                       label="uniform")
            ax.set_title(f"Rank uniformity across chains · {ranked_name}", loc="left",
                         fontsize=9)
            ax.set_xlabel("pooled rank")
            ax.legend(fontsize=6, frameon=False, ncol=min(n_chains + 1, 5))
        else:
            # Transitions are differences between CONSECUTIVE draws within a chain.
            # Diffing the chain-concatenated vector would manufacture one spurious
            # transition per chain boundary -- the gap between chain c's last draw and
            # chain c+1's first, which are independent -- and those land in the tail,
            # making the sampler look worse than it is on the panel that exists to judge it.
            energy2d = np.asarray(stats["energy"])
            if energy2d.ndim == 1:
                energy2d = energy2d[None, :]
            energy = energy2d.ravel()
            ax.hist(energy, bins=30, histtype="stepfilled", alpha=0.55,
                    label="energy")
            if energy2d.shape[-1] > 1:
                transitions = np.diff(energy2d, axis=-1).ravel()
                ax.hist(transitions, bins=30, histtype="step", lw=1.2,
                        label="energy transitions (within chain)")
            ax.set_title("Energy distribution vs transitions", loc="left", fontsize=9)
            ax.set_xlabel("energy")
            ax.legend(fontsize=6, frameon=False)

    notes = []
    if n_chains == 1:
        notes.append("One chain: r-hat is undefined and rank uniformity is uninformative.")
    if stats is not None and "diverging" in stats:
        n_div = int(np.asarray(stats["diverging"]).sum())
        total = n_chains * n_draws
        notes.append(f"{n_div}/{total} divergent transitions.")
    if notes:
        fig.text(0.01, 0.004, "  ".join(notes), fontsize=7, style="italic")

    fig.tight_layout()
    if path:
        fig.savefig(str(path), dpi=170, bbox_inches="tight")
    return fig


def plot_population_posteriors(groups, beta_max=10.0, path=None):
    """Population posterior per parameter, on the parameter's own scale.

    Drawn post-transform because plotting unconstrained coordinates under a bounded
    parameter's label misstates every value.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    draws = _draws(groups["posterior"]).reshape(-1, len(PARAM_NAMES))
    fig, axes = plt.subplots(
        len(PARAM_NAMES), 1, figsize=(4.6, 1.15 * len(PARAM_NAMES)), squeeze=False
    )
    for i, (ax, name) in enumerate(zip(axes[:, 0], PARAM_NAMES)):
        values = to_bounded(draws[:, i], i, beta_max)
        spread = float(values.max() - values.min())
        span = _param_xlim(name, beta_max)
        if spread > 0 and len(np.unique(values)) > 2:
            kde = gaussian_kde(values)
            # Evaluate across the WHOLE fixed axis, not just the data's own spread. A grid
            # padded around the draws would leave the density curve stopping partway across
            # a fixed panel, which reads as missing data rather than as negligible mass.
            if span is not None:
                grid = np.linspace(span[0], span[1], 200)
            else:
                grid = np.linspace(values.min() - 0.18 * spread,
                                   values.max() + 0.18 * spread, 200)
            ax.fill_between(grid, kde(grid), alpha=0.55, lw=0)
            ax.plot(grid, kde(grid), lw=1.0)
        else:
            ax.axvline(float(values.mean()), lw=1.4)
        if span is not None:
            ax.set_xlim(*span)
        ax.set_yticks([])
        ax.set_ylabel(name.replace("_", "\n"), rotation=0, ha="right",
                      va="center", fontsize=6, labelpad=4)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
    axes[-1, 0].set_xlabel("parameter value (model's own units)", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(str(path), dpi=170, bbox_inches="tight")
    return fig


def log_hb_figures(
    fit_paths, scores=None, wandb_run=None, output_dir=None,
    beta_max=10.0, references=None,
):
    """Build the figure set for one fit and log it to W&B.

    Parameters
    ----------
    fit_paths : mapping
        ``{"netcdf": ..., "sample_stats": ...}`` as returned by ``save_fit``.
    scores : mapping, optional
        Context-session count to held-out likelihood, plus an optional ``"matched"`` key.
        Drives the conditioning curve, which is the headline comparison figure.
    wandb_run : optional
        Run to log into. Figures are still written to ``output_dir`` without one.
    references : mapping, optional
        ``{name: (value, colour)}`` comparators for the conditioning curve.

    Returns
    -------
    dict
        Figure key to written path, for the keys that were produced.
    """
    output_dir = Path(output_dir or ".")
    written = {}

    # Everything from here is best-effort: this function's contract is that a completed
    # fit is never lost to a plotting problem, so *preparation* is guarded too, not only
    # the individual builds. Both steps below can fail on inputs the caller controls --
    # an unwritable or invalid output directory, and a missing or truncated artifact --
    # and either would otherwise propagate out of a function that promises not to.
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        groups = load_fit(fit_paths.get("netcdf"), fit_paths.get("sample_stats"))
    except Exception:
        logger.exception(
            "Could not prepare figures (output_dir=%r, netcdf=%r); none written.",
            str(output_dir), fit_paths.get("netcdf"),
        )
        return written

    specs = [
        ("hb/diagnostics", "hb_diagnostics.png",
         lambda p: plot_sampler_diagnostics(groups, path=p)),
        ("hb/population_posterior", "hb_population_posterior.png",
         lambda p: plot_population_posteriors(groups, beta_max=beta_max, path=p)),
    ]
    if scores:
        # Imported here rather than at module scope, and guarded: this pulls in
        # `hierarchical_bayes.__init__`, which imports `likelihood` and therefore jax. On
        # the Beaker image jax is present, but an offline regeneration environment with
        # only arviz/xarray installed would raise ImportError -- and losing the other two
        # figures to the conditioning curve's dependency would be the wrong trade.
        try:
            from aind_dynamic_foraging_models.hierarchical_bayes.plotting import (
                plot_conditioning_curve,
            )

            specs.append((
                "hb/conditioning_curve", "hb_conditioning_curve.png",
                lambda p: plot_conditioning_curve(
                    scores, references=references, path=p,
                    title="Held-out likelihood vs context sessions",
                ),
            ))
        except Exception:
            logger.exception("Could not prepare hb/conditioning_curve; skipping it.")

    for key, filename, build in specs:
        path = output_dir / filename
        try:
            build(str(path))
            written[key] = str(path)
        except Exception:
            # A figure is never worth losing a completed fit over.
            logger.exception("Could not build %s; continuing.", key)

    if wandb_run is not None and written:
        try:
            import wandb

            wandb_run.log({k: wandb.Image(v) for k, v in written.items()})
            logger.info("Logged %d HB figures: %s", len(written), ", ".join(written))
        except Exception:
            logger.exception("Could not log HB figures to W&B; files are in %s.", output_dir)

    return written
