from .solver import solve_poly, solve_all, solve_one
from .geode import (
    SeriesOptions,
    map_t_from_poly,
    S_eval,
    eval_S_via_geode,
    Q_cubic,
    solve_series,
)
from .series_solve import series_solve_all, series_one_root
from .hyper_catalan import (
    hyper_catalan_coefficient,
    evaluate_hyper_catalan,
    catalan_number,
    evaluate_quadratic_slice,
)
from .formal import FormalSeries
from .series import series_root
from .poly import Polynomial
from .numeric import (
    newton as newton_solve,
    aberth as aberth_solve,
    dk as dk_solve,
    companion_roots,
)
from .batched import (
    batched_poly_eval,
    batched_newton_step,
    torch_root_layer,
)
from .layerings import vertex_layering, edge_layering, face_layering
from .adapters import (
    charpoly_roots,
    ar_roots,
    invert_radial,
    ray_intersect_quartic,
)

__all__ = [
    "solve_poly",
    "solve_all",
    "solve_one",
    "SeriesOptions",
    "map_t_from_poly",
    "S_eval",
    "eval_S_via_geode",
    "Q_cubic",
    "solve_series",
    "series_solve_all",
    "series_one_root",
    "hyper_catalan_coefficient",
    "evaluate_hyper_catalan",
    "catalan_number",
    "evaluate_quadratic_slice",
    "batched_poly_eval",
    "batched_newton_step",
    "torch_root_layer",
    "FormalSeries",
    "series_root",
    "Polynomial",
    "newton_solve",
    "aberth_solve",
    "dk_solve",
    "companion_roots",
    "vertex_layering",
    "edge_layering",
    "face_layering",
    "charpoly_roots",
    "ar_roots",
    "invert_radial",
    "ray_intersect_quartic",
]
