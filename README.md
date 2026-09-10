# README

## Setup

### Prerequisites

1. Install `uv`
   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   or

   ```powershell
   # On Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Install `DVC`
   TBC

### Reproduction Setup

   ```bash
   git clone https://github.com/16acoy/sputum-colour
   cd sputum-colour
   uv sync --locked
   dvc pull
   dvc repro
   ```

### Development Setup

1. Clone this repo and enter the directory
   ```bash
   git clone https://github.com/16acoy/sputum-colour
   cd sputum-colour
   ```

2. Create and populate your environment file
   ```bash
   cp .env.sample .env
   ```
   Then set:
   - `GIT_USER` with your name
   - `GIT_EMAIL` with your email

3. Set up the Python dev environment

   ```bash
   uv sync --group dev
   ```
   The `dev` group installs developer tooling in addition to runtime dependencies.

4. Install and validate `pre-commit` hooks
   ```bash
    uv run --group dev pre-commit install
    uv run --group dev pre-commit autoupdate
    uv run --group dev pre-commit install-hooks
    uv run --group dev pre-commit run --all-files
   ```

5. Run styling and security checks if desired
   ```bash
   uv run ruff check .
   uv run ruff format .
   uv run bandit -r src
   ```
## Documentation

The main codebase implemented in this project repo falls within the `src/sputum_colour` and `analysis` directories.

`src/sputum_colour` contains package code with the bulk of underlying functionality for data loading, processing, and a range of desired analyses. There is a separate detailed `README` within the directory for further information.

`analysis` contains Jupyter notebooks for TBC, as described in the TBC report. There is a separate detailed `README` within the directory for further information.

The notebooks import and call most functions from the package to ensure reproducbility and reduce redundancy.