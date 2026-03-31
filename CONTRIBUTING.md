# Contributing to mfai

Thanks for contributing to `mfai`.

This repository is a Python package centered around deep-learning models, training utilities, tokenizers, and weather-oriented AI workflows. Contributions land best when they stay tightly scoped and match the existing package layout.

## Repository Layout

The main areas you are likely to touch are:

- `mfai/` for package code
- `tests/` for pytest-based coverage
- `examples/` for minimal runnable usage samples
- `doc/` for longer-form documentation assets
- `pyproject.toml` for dependencies, linting, typing, and test tooling

Within `mfai/`, the `pytorch/` subtree contains most of the model, metric, transform, and training-related code, while top-level modules such as `encoding.py`, `http.py`, and `tensorflow.py` cover more focused utilities and integrations.

## Good Contribution Areas

- bug fixes in package modules under `mfai/`
- targeted test additions under `tests/`
- documentation clarifications that match the current code and examples
- small improvements to examples, config files, or developer tooling
- typing, lint, or packaging fixes that align with `pyproject.toml`

## Before You Start

- Read [README.md](README.md) first to understand the supported model families and current workflow.
- Keep one PR to one logical change. Do not mix a docs rewrite, a model change, and unrelated cleanup in the same branch.
- Follow the existing style of the file you are editing instead of reformatting unrelated sections.

## Local Setup

The project metadata lives in `pyproject.toml`, and the repository already declares its dev tooling there.

A typical local workflow is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

If you also need the optional LLM-related dependencies, install:

```bash
pip install -e '.[dev,llm]'
```

## Fast Validation

Run the lightest checks that fit your change before opening a PR.

For a docs-only or metadata-only change:

```bash
git diff --check
```

For lightweight Python syntax validation:

```bash
python3 -m py_compile mfai/__init__.py mfai/http.py mfai/encoding.py
python3 -m py_compile examples/main_cli_dummy.py examples/train_test_cli_dummy.py
```

For focused test work:

```bash
pytest tests/test_namedtensors.py -q
```

For broader code-quality work, use the tooling already declared in `pyproject.toml`:

```bash
ruff check .
mypy mfai
pytest -q
```

It is fine to run a narrower subset if your PR only touches one module or one test area. Just mention the exact command you used in the PR body.

## Pull Request Expectations

Please include:

- a short summary of what changed
- why the change was needed
- the exact verification command or commands you ran
- any notable dependency or environment assumptions

Small, reviewable PRs are preferred over broad cleanup passes.

## Practical Guidelines

- Do not commit virtual environments, caches, model weights, or large generated artifacts.
- Keep examples and README snippets in sync with the current package API.
- If you change behavior, add or update a focused test when practical.
- If a change only affects one model family or one utility module, keep the PR scoped there instead of touching neighboring code for style reasons.
