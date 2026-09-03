# Training Record Rules

Purpose: keep every experiment record short, comparable, and easy to scan.

## 1. File naming

Use:

```text
iN-Architecture-Name-vVersion-Pass.md
iN-Architecture-Name-vVersion-Fail.md
```

- `N` is the next experiment number: `1` if no record exists, otherwise one more than the largest existing `iN`.
- Ignore `rules.md` when counting.
- Never reuse a number or overwrite an old run.
- Use hyphens in architecture names and keep the architecture version.
- `Pass` or `Fail` is the overall result for the primary goal, not one successful component.

## 2. Required record order

Keep the main body compact and use this order:

1. **Title and one-line decision** — state Pass/Fail and the main reason.
2. **Run metadata** — record, architecture, notebook/script, date if known, epochs, steps, stages, final learning rate, time, and source logs.
3. **Goal** — list the primary goal and important secondary goals.
4. **Key results** — use small tables for progress and final metrics.
5. **Findings** — short numbered points: what worked, what failed, and why.
6. **Next steps** — clear, actionable experiments.
7. **Technical appendix** — detailed losses, routing values, calibration values, derived values, and other specialist information. Keep technical detail at the bottom.
8. **Source note** — identify the raw logs or files.

Do not repeat the same metric in multiple sections unless the repetition is necessary for the decision.

## 3. Metadata requirements

Always record when available:

- Architecture and version.
- Record serial and notebook/script path.
- Training date; write `Not provided` if unknown.
- Epochs, steps, stages, final learning rate.
- Training time and its unit; preserve separate logger times when they differ.
- Dataset/configuration summary if known.
- Names of the raw metric and progress files.

Never guess missing metadata. Use `Not provided` or `Not available`.

## 4. Metric requirements

Choose metrics that match the architecture. For a real/fake detector, include:

- Training AUC and evaluation/test AUC.
- Worst/selection AUC, mean AUC, and clean AUC.
- AUC/AP/EER for each important robustness condition.
- Default-threshold confusion counts and accuracy, recall, F1, or balanced accuracy.
- Calibrated threshold results separately from default-threshold results.
- Localization pixel AUC, IoU, and mask F1 when applicable.
- Routing shares and dead experts when routing is used.

Use compact rounded tables in the main body. Put extra precision and specialist metrics in the technical appendix. Do not paste complete raw JSON into the record.

## 5. Findings and decision rules

- Judge the primary goal first; secondary metrics cannot hide primary failure.
- Use a project-defined acceptance threshold when one exists and state it.
- If no threshold exists, do not invent a precise one. Explain the evidence.
- AUC near `0.50` is chance-level.
- Slightly above-chance evaluation AUC is not enough to claim a useful detector.
- A large train/evaluation gap is an overfitting or distribution-mismatch finding.
- Predicting one class at the default threshold is a deployment-readiness failure.
- Good localization or another auxiliary result does not make the full architecture Pass when the primary task fails.
- Mark `Pass` only when the primary task is useful on evaluation data and no critical issue is hidden.
- Mark `Fail` when the primary task is not useful or a critical issue blocks the intended use; preserve promising component results in Findings.
- Always report default and calibrated/post-hoc results separately.

## 6. Compact writing rules

- Put the conclusion near the top.
- Prefer tables and numbered findings over long explanations.
- Use plain language first; put formulas, loss names, routing diagnostics, and implementation details in the appendix.
- Report actual values; do not fabricate or silently remove poor results.
- Label every result as training, evaluation, test, clean, robustness, default-threshold, or calibrated.
- Preserve contradictory evidence, such as perfect training performance with weak evaluation performance.
- Use one short sentence for the meaning of each important result.

## 7. Creation checklist

1. Inspect existing records and select the next `iN`.
2. Read the architecture name/version and collect actual logs.
3. Create the record using the required order.
4. Decide Pass/Fail from the primary goal.
5. Keep the main body compact; move technical detail to the appendix.
6. Read the file back and verify the title, metadata, key metrics, findings, decision, and source note.
