# AAAI-27 ABER Submission Checklist

Last updated: 2026-07-27

## Upload Files

1. Main paper:
   `paper/aaai_2027/aaai27/main_aber_aaai27.pdf` (8 pages: 7 body + references).
2. Reproducibility checklist:
   `paper/aaai_2027/aaai27/repro_checklist_aber_v1.pdf` (2 pages).
3. Supplement:
   `paper/aaai_2027/aaai27/supplementary_aber_v1.pdf` (10 pages).
4. Anonymous code/data archive:
   rebuild as `paper/aaai_2027/anonymous_code_data_aber_v7.zip` before upload.
   The existing V6 archive predates the complete ABER-LR cross-task matrix and
   must not be uploaded as the final evidence package.

Do not upload internal audit Markdown files, historical method figures,
hardware logs, or third-party repositories.

## Clean Build

From `paper/aaai_2027/aaai27`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main_aber_aaai27.tex
bibtex main_aber_aaai27
pdflatex -interaction=nonstopmode -halt-on-error main_aber_aaai27.tex
pdflatex -interaction=nonstopmode -halt-on-error main_aber_aaai27.tex

pdflatex -interaction=nonstopmode -halt-on-error supplementary_aber_v1.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary_aber_v1.tex

pdflatex -interaction=nonstopmode -halt-on-error repro_checklist_aber_v1.tex
pdflatex -interaction=nonstopmode -halt-on-error repro_checklist_aber_v1.tex
```

`latexmk` is not installed in the current environment; the explicit
pdflatex/BibTeX sequence above is the verified build.

## Final Checks

- Main is US Letter and at most 9 pages.
- All technical content is within the 7-page body limit.
- Checklist is separate from the main PDF.
- Author/affiliation remain anonymous.
- No overfull boxes, undefined references, fatal errors, or emergency stops.
- All fonts are embedded and no Type 3 font appears.
- Supplement title and text use the current ABER/ABER-LR method identity.
- Code archive contains current ABER code, every aggregate and raw evidence
  file, the paired checkpoint, generation/validation scripts, and focused tests.
- Archive excludes old method results and all identity-bearing paths/metadata.

## Evidence Allowed in the Submission

- `data/aber_property_tests.json`
- `data/certified_diffdrive_benchmark.json`
- `data/safetygym_aber_matrix.json`
- `data/aber_component_ablation.json`
- `data/aber_component_ablation_trials.jsonl.gz`
- `data/aber_runtime_scaling.json`
- `data/aber_runtime_scaling_trials.jsonl.gz`
- `data/safetygym_aber_matrix_retrained.json`
- `data/aber_paired_matrix.json`
- `data/aber_paired_matrix_verification.json`
- `ABER_LATCHED_RESTORE_CROSS_TASK_MULTISEED_V1_PROTOCOL.json`
- `results/aaai27_aber_latched_restore_cross_task_v1/cross_task_audit.json`
- `results/aaai27_aber_latched_restore_cross_task_v1/full_30_checkpoint_aggregate.json`
- strict and ABER-LR raw paired rollouts for all 30 checkpoints, including
  checkpoint, initial-state, and filter-off identity hashes
- 30 paired checkpoints, per-cell episode/step raw files, and current scripts in
- `ABER_SAFETYGYM_CONTRACT_V1_PROTOCOL.json`
- `data/aber_safetygym_contract_v1.json` plus 30 raw substep trace files
  `METHOD_CODE_ALIGNMENT.md`

Any additional result requires a current-code rerun plus a source/config/data
manifest. Missing cells must cause failure; never impute or reconstruct them.
