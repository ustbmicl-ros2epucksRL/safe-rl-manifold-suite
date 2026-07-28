# Figure directories

- `main/`: the single figure included by `aaai27/main_aber_aaai27.tex`.
- `supplement/`: the five figures included by
  `aaai27/supplementary_aber_v1.tex` and its generated inputs.
- `backup/`: historical or currently unused figures and media.  Nothing in
  this directory is referenced by the current paper build.
- `scripts/`: plotting sources for the current ABER-LR figures.

Build one current figure with, for example:

```bash
make -C paper/aaai_2027/figures main
make -C paper/aaai_2027/figures recovery
make -C paper/aaai_2027/figures falsification
```

`main/aber_lr_falsification_chain.pdf` is a completed standalone candidate for
the main paper; it is not included by the LaTeX source until the main-paper
layout decision is made.

Rebuild and verify all current figures with:

```bash
make -C paper/aaai_2027/figures verify-active
```
