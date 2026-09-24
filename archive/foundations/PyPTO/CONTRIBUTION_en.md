# Contribution Guide

This project welcomes developers to experience and participate in the community contribution. Before participating in community contributions, refer to [cann-community](https://gitcode.com/cann/community) to understand the code of conduct, sign the CLA agreement, and learn about the source repository contribution process.

When developers prepare local code and submit a PR, they need to focus on the following points:

1. When submitting a PR, fill in the PR template carefully with the business background, purpose, solution, and other information.
2. If your modification is not a simple bug fix but involves adding new features, new interfaces, new configuration parameters, or modifying code flow, discuss the solution through an Issue first to avoid your code being rejected. If you are unsure whether this modification qualifies as a "simple bug fix," you can also submit an Issue for solution discussion.

Developer contribution scenarios mainly include:

- Bug Fix

  If you find a bug in this project and want to fix it, create a new Issue for feedback and tracking.

  Follow the [Submit Issue/Handle Issue Tasks](https://gitcode.com/cann/community#Submit-Issue-Handle-Issue-Tasks) guide to create a `Bug-Report` Issue to describe the bug. Then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself for processing.

- Code Optimization

  If you have ideas for generalization enhancement or performance optimization for some API implementations in this project and want to implement these optimizations, contribute by optimizing the API.

  Follow the [Submit Issue/Handle Issue Tasks](https://gitcode.com/cann/community#Submit-Issue-Handle-Issue-Tasks) guide to create a `Requirement` Issue to describe the optimization points and provide your design solution. Then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself for tracking and optimization.

- Documentation Correction

  If you find documentation errors in this project, create a new Issue for feedback and correction.

  Follow the [Submit Issue/Handle Issue Tasks](https://gitcode.com/cann/community#Submit-Issue-Handle-Issue-Tasks) guide to create a `Documentation` Issue to point out the corresponding document problem. Then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself for correcting the document description.

- Help Resolve Others Issues

  If you have a suitable solution to problems others encounter in the community, leave a comment in the Issue to help them solve the problem and pain point, and jointly optimize ease of use.

  If the corresponding Issue requires code modification, enter "/assign" or "/assign @yourself" in the Issue comment box to assign the Issue to yourself for tracking and solving the problem.

---

## Code Style and Pre-commit Checks

This project uses [pre-commit](https://pre-commit.com/) to automatically enforce code style checks and formatting before each commit, ensuring all contributed code follows consistent coding standards. Developers **must** pass pre-commit checks before submitting code.

### Installation

```bash
pip install pre-commit
pre-commit install
```

Once installed, every `git commit` will automatically trigger checks. If any check fails, the commit will be blocked.

### Checks Overview

The configuration is in `.pre-commit-config.yaml` and includes:

| Check | Tool | Description |
|-------|------|-------------|
| Trailing whitespace / EOF newline | pre-commit-hooks | Strip trailing whitespace, ensure files end with newline |
| YAML / JSON validation | pre-commit-hooks | Validate config file syntax |
| Large files / private keys / merge conflicts | pre-commit-hooks | Prevent accidental binary/key commits, check merge conflict markers |
| C++ formatting | clang-format (v18) | Format C/C++/ASC files per `.clang-format` |
| Python linting | ruff check (v0.14) | Static checks for E/W/F/I/N rule groups, auto-fix where possible |
| Python slice colon spacing | local script | Check spacing around slice colons (covers ruff blind spots) |
| Spell check | codespell | Detect common spelling errors |

### Python Code Standards (ruff)

Ruff is configured in `pyproject.toml` under `[tool.ruff]`. Enabled rule groups:

- **E / W** — pycodestyle errors and warnings (e.g., E711 comparisons, E741 ambiguous names)
- **F** — pyflakes (e.g., F401 unused imports, F841 unused variables, F821 undefined names)
- **I** — isort import sorting
- **N** — pep8-naming conventions (e.g., N802 function names, N806 variable names, N818 exception names)

`E501` (line length) is ignored; the line length limit is **120** characters.

### Running Checks Manually

```bash
# Run all checks on all files
pre-commit run --all-files

# Run only ruff check
pre-commit run ruff-check --all-files

# Run only C++ formatting check
pre-commit run clang-format --all-files

# Run only slice colon spacing check
pre-commit run slice-colon-spacing --all-files

# Run only spell check
pre-commit run codespell --all-files
```
