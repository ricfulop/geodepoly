# GeodePoly v0.2.4

Date: 2025-08-25

## Highlights
- Non-finance gap analysis completed: new series APIs, Geode utilities, layerings, adapters.
- New fast-path: `method="hybrid-cubic"` for quick warm-starts using `Q_cubic`.
- Bring/Eisenstein series (`bring_radical_series`) and Lagrange reversion (`series_reversion_coeffs`).
- Documentation expanded with Geode, Layerings (OEIS note), Cubic/Quintic, and Adapters pages.

## Changes
- Added `geodepoly/geode.py`: `SeriesOptions`, `map_t_from_poly`, `S_eval`, `eval_S_via_geode`, `Q_cubic`, `solve_series`, `series_reversion_coeffs`, `bring_radical_series`.
- Added `geodepoly/layerings.py`: `vertex_layering`, `edge_layering`, `face_layering`.
- Added adapters: controls (`charpoly_roots`), signals (`ar_roots`), vision (`invert_radial`), geometry (`ray_intersect_quartic`).
- Solver: added `method="hybrid-cubic"`; `solve_series` uses robust series seeding.
- Tests: added coverage for Geode identity, layerings, cubic/bootstrapping, Bring, and reversion.
- Docs: new pages and nav updates; mkdocs build verified.

## Compatibility
- No breaking API removals. New optional method in `solve_all`.

## Thanks
- Contributors and users trying early features—please report issues.
