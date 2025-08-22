from geodepoly import (
    hyper_catalan_coefficient,
    evaluate_hyper_catalan,
    catalan_number,
    evaluate_quadratic_slice,
)


def test_hyper_catalan_zero_multiindex():
    assert hyper_catalan_coefficient({}) == 1


def test_hyper_catalan_simple_terms():
    # m2=1 → coef = 2! / ( (1+1)! * 1! ) = 2 / 2 = 1
    assert hyper_catalan_coefficient({2: 1}) == 1
    # m3=1 → 3! / ( (1+2)! * 1! ) = 6 / 6 = 1
    assert hyper_catalan_coefficient({3: 1}) == 1
    # m2=2 → 4! / ( (1+2)! * 2! ) = 24 / (6*2) = 2
    assert hyper_catalan_coefficient({2: 2}) == 2


def test_quadratic_slice_matches_catalan_series():
    # S with only t2 should generate Catalan coefficients in alpha(0)=1 branch
    t2 = 0.01
    approx = evaluate_quadratic_slice(t2, max_weight=12)
    # Catalan series: sum_{n>=0} C_n t2^n
    partial = 0.0
    for n in range(0, 7):
        partial += catalan_number(n) * (t2 ** n)
    assert abs(approx - partial) < 1e-6


def test_multivariate_truncation():
    val = evaluate_hyper_catalan({2: 0.1, 3: 0.05}, max_weight=5)
    assert isinstance(val, complex)

