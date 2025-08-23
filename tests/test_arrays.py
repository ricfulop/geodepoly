from geodepoly.hyper_catalan import bi_tri_array, hyper_catalan_coefficient


def test_bi_tri_small_values():
    A = bi_tri_array(3, 2)
    # spot-check a few entries against direct hyper_catalan_coefficient
    assert A[0][0] == hyper_catalan_coefficient({})
    assert A[1][0] == hyper_catalan_coefficient({2: 1})
    assert A[0][1] == hyper_catalan_coefficient({3: 1})
    assert A[2][1] == hyper_catalan_coefficient({2: 2, 3: 1})


