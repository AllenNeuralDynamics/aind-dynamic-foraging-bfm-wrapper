# Global AGENTS.md

## Explaining and Reporting Style

Source: Karpathy's advice to ask for explanations in
[ASD-STE100](https://www.asd-ste100.org/) (Simplified Technical English), at about 80%
strictness.

**In conversation** (more freedom than reports; aim to teach):
- **Plain language, roughly 80% of ASD-STE100.** Short sentences, active voice, no filler.
  This is a guide, not a hard rule. Use longer sentences when they make an idea clearer.
- **Explain as a teacher would.** Do not assume the reader knows every technique. The first
  time a specialised term appears, give its basic idea in a sentence or two. This applies
  to statistical methods (for example hierarchical Bayes, posterior predictive checks or
  bootstrapping), model internals and training mechanics. Say what the term means, why it
  is used here, and what to watch for. After that, use the term without repeating the
  explanation.
- **Show, do not only tell.** Explain with a diagram where one helps. For a long or
  structured explanation, prefer an HTML page over a wall of text.

**In reports:**
- **Plain language, about 80% of ASD-STE100.** Short sentences, one idea per sentence,
  active voice, no filler.
- **Order of preference: figure > table > narrative.** A figure beats a table. A table
  beats prose. Use narrative only for what a figure or table cannot carry.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

The four-principle backbone (Think Before Coding, Simplicity First, Surgical Changes, Goal-Driven Execution) is adapted from [multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills), which distills [Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876) on LLM coding pitfalls. HPC-specific rules and the commit-message convention below are local additions.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them instead of picking silently.
- If a simpler approach exists, say so.
- If something is unclear, stop and ask.

## 2. Simplicity First

Write the minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No configurability that was not requested.
- No error handling for impossible scenarios.
- If the solution is overcomplicated, simplify it.

## 3. Surgical Changes

Touch only what is needed. Clean up only your own mess.

When editing existing code:
- Do not improve adjacent code, comments, or formatting unless required.
- Do not refactor unrelated code.
- Match existing style.
- If you find unrelated dead code, mention it but do not delete it.

When your changes create orphans:
- Remove imports, variables, or functions made unused by your change.
- Do not remove pre-existing dead code unless asked.

Test: Every changed line should trace directly to the request.

## 4. Goal-Driven Execution

Define success criteria and verify.

Transform tasks into verifiable goals:
- Add validation -> write failing tests for invalid inputs, then make them pass.
- Fix a bug -> write a reproducing test, then make it pass.
- Refactor -> ensure tests pass before and after.

For multi-step tasks, use a brief plan:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

These guidelines are working when diffs contain fewer unnecessary changes, solutions are simpler, and clarifications happen before implementation.

## 5. HPC Execution Safety

Never run computation-intensive work on the login node (where the agent runs).

- Always use `srun` or `sbatch` for heavy workloads.
- This includes training jobs, sweeps, and tests.

## 6. Semantic Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format for every commit:

```text
<type>(<optional scope>): <short imperative summary>

<optional body explaining what and why, wrapped at ~72 chars>

<optional footer, e.g. "Refs #123" or "BREAKING CHANGE: ...">
```

Allowed `<type>` values:

- `feat` — new user-visible feature
- `fix` — bug fix
- `docs` — documentation only
- `refactor` — code change that neither fixes a bug nor adds a feature
- `perf` — performance improvement
- `test` — add or fix tests
- `build` — build system, dependencies, or packaging
- `ci` — CI configuration or scripts
- `chore` — maintenance, tooling, or housekeeping with no src/test impact
- `revert` — revert a prior commit

Rules:

- Summary line in the imperative mood, no trailing period, <= 72 chars.
- One logical change per commit; split unrelated changes into separate commits.
- Use `<scope>` for the affected area when helpful (e.g. `feat(launcher): ...`, `docs(readme): ...`).
- Mark breaking changes with `!` after the type/scope (e.g. `feat(api)!: ...`) and a `BREAKING CHANGE:` footer.
- Body explains the motivation and any non-obvious consequences; don't restate the diff.

## 7. Project Skills & Living Docs

- The **Agent Skills pack** for this two-repo project lives in the dispatcher:
  `aind-dynamic-foraging-bfm-dispatcher/aind-behavior-foundation-model-skills/`. It is the
  canonical home for cross-cutting operational knowledge — Beaker/HPC launching,
  study conventions, post-hoc reporting, and this repo's runtime distilled in the
  `wrapper-runtime` skill. It is auto-enabled here via the checked-in
  `.claude/settings.json` (Claude Code plugin marketplace).
- This repo's **living documents stay canonical** for code-coupled reference:
  `code/TRAINING.md` (read §1.5 "Run lifecycle & key switches" first) and
  `code/POST_TRAINING_ANALYSIS.md`. When you add or change a feature, update the
  relevant guide + its Changelog — the skills defer to these files.
- Never squash-merge a PR (`gh pr merge <n> --merge`) — preserve per-commit history.
