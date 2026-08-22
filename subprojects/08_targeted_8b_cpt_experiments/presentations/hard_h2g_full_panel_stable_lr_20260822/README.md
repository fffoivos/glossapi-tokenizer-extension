# Hard H2G full-panel + stable-LR report

This directory is the reviewable report package for
`HARD_H2G_FULL_PANEL_SCORING_AND_NO_DECAY_BRANCH_PLAN_20260822.md`.

Primary output:

- `HARD_H2G_FULL_PANEL_AND_STABLE_LR_20260822.html`

Build order:

```bash
python3 build_analysis.py
python3 build_report.py
```

The analysis builder fails closed unless it finds:

- all 17 full-public FP32 checkpoints from the original 8B trajectory;
- the four paired stable-LR checkpoints 2,618 / 2,856 / 3,094 / 3,218;
- all nine paired validation panels;
- every stable optimizer update at exactly `5.5e-5` with zero skipped and
  zero non-finite updates;
- the frozen legacy-BF16 replication receipt.

QA renders the complete page at desktop and narrow widths, inspects both
images, and then runs:

```bash
python3 verify_report.py --visual-inspection-passed
```

The final post-build delivery step is the repository-independent Firefox
helper required by the academic HTML report skill.
