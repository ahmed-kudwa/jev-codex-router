# Jev-align workflow

`jev-align` is an optional offline alignment tool for this router. It is not
part of the live request path and it does not replace the local Hive classifier
or Jev. The router first exports a bounded, redacted trace dataset; a human
labels uncertain route classes and effort levels; `jeva` can then use those
labels to propose an improved definition through GEPA.

This keeps production routing fast and makes every policy change reviewable.
The router never accepts a GEPA proposal automatically.

## Validate the tool

The upstream project is alpha software and its repository currently contains a
license declaration mismatch: `LICENSE` says MIT while `pyproject.toml`
declares Apache-2.0. This router does not vendor or redistribute its code, so
the mismatch is not copied into this repository. Re-check it before packaging
the upstream project for redistribution.

Use a disposable environment for validation:

```sh
uv run --extra dev --project /path/to/jev-align pytest
```

The upstream test suite is expected to pass before using its optimizer. The
router integration below only depends on Python's standard library.

## Export a dataset

The exporter reads the local decision trace and writes a CSV with bounded,
redacted fields:

```sh
python3 scripts/jev_align_export.py
```

The default output is:

```text
~/.codex/codex-router/jev-align/routing-dataset.csv
```

The Hive dashboard endpoint (`/hive/status`) reports whether this dataset
exists, its row count, size, and last refresh time under the `alignment` field.
The supervised router refreshes the export on its hourly maintenance tick. This
is metadata only and does not expose the dataset contents.

The export contains route metadata, context-size buckets, transport status,
and an empty pair of columns for human labels. It removes common API-key,
Bearer-token, secret, and host-path patterns. It never includes tool output or
the full request history. The output is local state and must not be committed.

To use a different trace or limit the sample:

```sh
python3 scripts/jev_align_export.py \
  --source ~/.codex/codex-router/jev-router-live.jsonl \
  --output ~/.codex/codex-router/jev-align/routing-dataset.csv \
  --limit 1000
```

## Label and optimize

Install `jev-align` in a separate environment, then run the guided optimizer:

```sh
jeva optimize ~/.codex/codex-router/jev-align/routing-dataset.csv \
  --question "Choose the best route class for this coding-agent step" \
  --column task_summary \
  --column step_type \
  --column context_chars \
  --column context_items \
  --class "mechanical=Prepared command, read, check, or simple continuation" \
  --class "routine=Small, clear implementation task" \
  --class "hard_engineering=Multi-file debugging or design task" \
  --class "recovery=Step after a failure" \
  --class "review=Independent review or regression verification" \
  --class "consequential=Security, production, migration, or destructive change" \
  --class "other=Does not fit another route class"
```

Use the fixed route classes when the CLI asks for labels:

- `mechanical`: prepared commands, reads, checks, and simple continuations
- `routine`: small, clear implementation work
- `hard_engineering`: multi-file debugging or design work
- `recovery`: a step following a failure
- `review`: independent review or regression verification
- `consequential`: security, production, migration, or destructive changes
- `other`: anything that does not fit the above

The proposed definition is an input to review. Promote it only after a held-out
evaluation shows a meaningful quality improvement without a cost or escalation
regression. A promotion should be implemented as a small source-controlled Hive
policy change, then exercised in shadow mode before it can affect live routing.

## Safety boundaries

- `jev-align` does not run in the request path.
- The exporter does not call a provider and never reads API keys.
- A transport `200` is not treated as semantic success; human labels are
  required for alignment.
- GEPA output remains a proposal until it passes the router's normal review,
  test, and rollback process.
- Keep the local logistic learner in shadow mode unless a later held-out
  evaluation authorizes a bounded promotion.
