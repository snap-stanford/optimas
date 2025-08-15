# Contributing to Optimas

Thank you for your interest in contributing to Optimas! We welcome contributions from the community.

## How to Contribute
- **Bug Reports & Feature Requests:** Please use [GitHub Issues](https://github.com/stanfordnlp/optimas/issues).
- **Pull Requests:** Fork the repo, create a feature branch, and submit a pull request (PR) with a clear description.
- **Discussions:** For design or usage questions, open a GitHub Discussion or join our community chat.

## Code Style & Quality
- Follow [PEP 8](https://peps.python.org/pep-0008/) and [PEP 621](https://peps.python.org/pep-0621/) standards.
- All code must pass [ruff](https://docs.astral.sh/ruff/), [black](https://black.readthedocs.io/en/stable/), and [isort](https://pycqa.github.io/isort/) checks.
- Use type hints where possible.

## Development Setup

### Using [uv](https://github.com/astral-sh/uv) (recommended)
[uv](https://github.com/astral-sh/uv) is a fast, modern Python package and dependency manager. You can use it for all dependency management and lockfile generation in Optimas.

1. Install uv:
   ```bash
   pip install uv
   ```
2. Install all dependencies (including dev tools):
   ```bash
   uv pip install .[dev]
   ```
3. (Optional) Generate a lock file for reproducibility:
   ```bash
   uv pip compile pyproject.toml > uv.lock
   ```
4. Run tests:
   ```bash
   pytest
   ```

### Traditional pip (alternative)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # or, for full dev setup:
   pip install .[dev]
   ```
2. Run tests:
   ```bash
   pytest
   ```

## Pre-commit Hooks
- We use [pre-commit](https://pre-commit.com/) to enforce code style and quality.
- Hooks: ruff, black, isort, end-of-file-fixer.
- Run `pre-commit run --all-files` before pushing.

## Submitting a Pull Request
- Ensure your branch is up to date with `main`.
- All tests and pre-commit hooks must pass.
- Add/Update documentation and tests as needed.
- Add a changelog entry in `CHANGELOG.md` if your PR is user-facing.

## License
By contributing, you agree that your contributions will be licensed under the MIT License.
