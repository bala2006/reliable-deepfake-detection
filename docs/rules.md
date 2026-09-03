# Docs Folder Rules

Purpose: store the reference/design document for each architecture iteration.

## Naming

Use:

```text
iN-Architecture-Name-vVersion.docx
```

For this project:

```text
i1-Stable-RouteNet-v4.docx
```

`iN` is the experiment iteration and must match the architecture file, notebook, and result record.

## Rules

- Keep one source document per architecture iteration.
- Never overwrite an earlier iteration; create a new `iN` file.
- Preserve the original document format and content unless a revision is explicitly requested.
- Use the same architecture name and version in the filename and document title when possible.
- Keep design/reference material here; keep compact measured results in `record/`.
- If the document is revised without a new training iteration, note the revision separately instead of changing `iN`.
