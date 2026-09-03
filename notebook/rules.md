# Notebook Folder Rules

Purpose: store executable training/evaluation notebooks for each architecture iteration.

## Naming

Use:

```text
iN-Architecture-Name-vVersion.ipynb
```

For this project:

```text
i1-Stable-RouteNet-v4.ipynb
```

`iN` is the experiment iteration and must match the architecture, document, and result record.

## Required notebook identity

The first markdown cell or run metadata should identify:

- Architecture and version.
- Iteration `iN`.
- Dataset/configuration.
- Training epochs and main hyperparameters.
- Output metric locations, if available.

## Rules

- Never overwrite an earlier iteration notebook; create a new `iN` file.
- Preserve executed outputs when they are evidence for a recorded run.
- Keep code and run configuration together so the result can be reproduced.
- Record changes from the previous iteration in the first markdown cell or metadata.
- Link or refer to the matching architecture, document, and result record.
- Store measured conclusions in `record/`, not only in notebook output.
