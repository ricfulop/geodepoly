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

## Commit style
- Use clear, scoped messages (`feat:`, `fix:`, `docs:`, `bench:`).
- Add tests when possible.

## Releasing
- Bump version in `pyproject.toml` and update `CHANGELOG.md`.
- Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`.
- Create a GitHub Release and attach built artifacts (sdist/wheel).
