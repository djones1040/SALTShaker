# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

SALTShaker is the training framework for the SALT3 spectrophotometric model of
Type Ia supernovae. It fits model components (M0, M1, color law, errors) to
combined photometry and spectroscopy, using a JAX backend for optimization.
Published on PyPI as `saltshaker-sn`.

## Setup and commands

```bash
pip install -r requirements.txt   # pinned versions used by CI (Python 3.10)
pip install -e .

# Run the test suite (CI runs exactly this from the repo root)
export PYTHONPATH=.
pytest

# Run a single test
pytest saltshaker/tests/test_likelihood.py

# Train a model
trainsalt -c path/to/training.conf
trainsalt -g                       # download public training data to $OSTRICH_INITFILES

# End-to-end pipeline (sim -> train -> fit -> cosmology, uses SNANA)
runpipe ...
```

Tests reference paths like `testdata/test.conf`, so always run pytest from the
repository root.

## Layout

- `saltshaker/training/` — core training code
  - `TrainSALT.py` — `TrainSALT` and `RunTraining` (CLI entry, config parsing, main loop)
  - `saltresids.py` — residuals / likelihood calculation (performance-critical, JAX)
  - `datamodels.py` — photometric and spectroscopic data model classes
  - `colorlaw.py`, `priors.py`, `constraints.py`, `init_templates.py`
  - `optimizers/` — `gaussnewton.py`, `gradientdescent.py`
- `saltshaker/config/` — config parsing (`configparsing.py`), default `training.conf`
- `saltshaker/util/` — readers for SNANA-format data, kcor files, etc.
- `saltshaker/pipeline/` — `runpipe` pipeline orchestration
- `saltshaker/validation/`, `saltshaker/plotting/` — model validation and figures
- `saltshaker/scripts/` — installed CLI scripts (`trainsalt`, `runpipe`)
- `saltshaker/initfiles/`, `saltshaker/data/kcor/` — packaged templates, filters, kcor FITS
- `testdata/` — small datasets and configs used by the tests
- `docs/` — Sphinx docs built on Read the Docs

## Configuration

Training configs are INI files with sections `[trainingparams]`, `[modelparams]`,
`[priors]`, `[bounds]`, `[init_offsets]`. New options must be registered in
`saltshaker/config/configparsing.py` and given a default in
`saltshaker/config/training.conf`; document user-facing options in `docs/`.

## Conventions

- Keep code JAX-compatible in the likelihood/residual path: avoid Python-side
  loops over SNe and in-place NumPy mutation inside jitted functions.
- Numerical changes to the likelihood, priors, or color law can change trained
  models used in published cosmology analyses. Flag these clearly in PRs and
  run `test_likelihood.py`, `test_colorlaw.py`, and `test_specrecal.py`.
- Don't modify files in `saltshaker/initfiles/` or `saltshaker/data/kcor/`
  unless the task is explicitly about them.
- Python 3.10 is the CI target; dependency pins live in `requirements.txt`,
  minimum versions in `setup.py`. Update both if you change a dependency.
- Note: `.gitignore` ignores all dotfiles (`.*`), so new files under `.github/`
  or `.claude/` need `git add -f`.
