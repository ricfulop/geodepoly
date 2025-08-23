# Contributing to geodepoly

Thanks for your interest!

## Dev setup
```bash
git clone https://github.com/<your-username>/geodepoly.git
cd geodepoly
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

### Pre-commit hooks
```bash
pip install pre-commit
pre-commit install
# run on all files
pre-commit run --all-files
```

## Commit style
- Use clear, scoped messages (`feat:`, `fix:`, `docs:`, `bench:`).
- Add tests when possible.

## Releasing
- Bump version in `pyproject.toml` and update `CHANGELOG.md`.
- TestPyPI: `git tag test-X.Y.Z && git push origin test-X.Y.Z` (requires `TEST_PYPI_API_TOKEN`).
- PyPI: `git tag vX.Y.Z && git push origin vX.Y.Z` (requires `PYPI_API_TOKEN`).
- The publish workflow builds and uploads automatically based on the tag prefix (`test-` vs `v`).
- Create a GitHub Release with notes.
