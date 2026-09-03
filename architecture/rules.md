# Architecture Folder Rules

Purpose: store one clear architecture/design document for each experiment iteration.

## Naming

Use:

```text
iN-Architecture-Name-vVersion.md
```

For this project:

```text
i1-Stable-RouteNet-v4.md
```

`iN` is the experiment iteration and must match the notebook, document, and result record.

## Required content

Keep each architecture document organized and easy to scan:

1. Purpose and plain-language goal.
2. Main architecture components and data flow.
3. Training data, split, and evaluation setup when known.
4. Training stages or curriculum.
5. Evaluation metrics and selection rule.
6. Important assumptions, risks, and changes from the previous iteration.
7. Technical details after the plain-language summary.

## Rules

- Never overwrite an earlier iteration; create a new `iN` file.
- Use actual architecture details; mark unknown information as `Not provided`.
- Keep the version in the filename and document title.
- Link or refer to the matching notebook, document, and result record using the same `iN` identity.
- Keep the document focused on design and intended behavior; place measured results in `record/`.
