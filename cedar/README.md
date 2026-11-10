# Cedar image

A GPU image of the wrapper for [Cedar](https://cedarai.org), the Allen Institute's compute
environment. It has the same software as the Beaker image (`../beaker/`): dispatcher,
`wrapper[gpu]` and `models[bayes]`, cloned from GitHub at full commit SHAs.

The Beaker image cannot run on Cedar, for two reasons:

| Cedar needs | So this image |
|---|---|
| The `cedar-session` startup command, which runs the SSH server and logs you in as `cedar` | is built `FROM` `cedar-session`, with no `ENTRYPOINT`, `CMD` or final `USER` |
| Nothing under `/workspace`, `/home`, `/homedir`, `/scratch` or `/output`, where Cedar mounts storage | keeps the repos in `/opt/bfm` and a venv in `/opt/env`, owned by `cedar` |

## Build and push

On a computer with Docker (a Mac is fine), signed in to Cedar (`cedar auth whoami`):

```sh
bash cedar/build_and_push.sh                      # main of all three repos
bash cedar/build_and_push.sh --wrapper-ref <sha>  # a specific wrapper commit
```

The script resolves each ref to a full SHA, pulls `cedar-session` for `linux/amd64`, builds,
and pushes `dfbfm-wrapper:<YYYYMMDD>-w<wrapper sha7>` to the workspace
`neural-dynamics-playground`. It stops if that tag already exists, so a tag always names one
image. `--help` lists the options. Add a line to `../beaker/BUILD_LOG.md` for each push.

## Smoke test

The script prints the commands. In short: start a session from the image on a GPU cluster,
then run

```sh
ssh cedar-<workspace>-<session> 'bash /opt/bfm/aind-dynamic-foraging-bfm-wrapper/cedar/smoke.sh'
```

`smoke.sh` runs the checks in `../beaker/smoke.yaml` (JAX on the GPU, the bayes extra, Hydra,
two training steps, no W&B), plus a read of database snapshot `20260603`. It runs through
`../beaker/entrypoint.sh`, pinned to the commits already in the image. It ends with
`SMOKE OK`.

## In a session

- `python` is `/opt/env/bin/python`. The repos are under `$BFM_ROOT` (`/opt/bfm`).
- To run other code, use the entrypoint as on Beaker. It fetches the refs you name, then
  runs your command:
  `WRAPPER_REF=<sha> bash $BFM_ROOT/aind-dynamic-foraging-bfm-wrapper/beaker/entrypoint.sh python -m run_hpc ...`
- Put working files on `/scratch` (fast local disk, lost at session end). Keep results in
  `$CEDAR_OUTPUTS_DIR` or `/workspace`. On GCP clusters those two are Cloud Storage buckets,
  so do not put a venv or a git checkout there.
- W&B: store the key once as a workspace secret (`cedar secret`), then pass
  `--secret <name>=WANDB_API_KEY` to `cedar session create`.
