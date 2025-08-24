from geodepoly.geode_conv import geode_convolution_dict


def test_geode_convolution_basic():
    # Univariate sanity: (1 + t2) * (1 + 2 t2) = 1 + 3 t2 + 2 t2^2
    A = {(): 1.0 + 0.0j, (1,): 1.0 + 0.0j}
    B = {(): 1.0 + 0.0j, (1,): 2.0 + 0.0j}
    C = geode_convolution_dict(A, B, num_vars=1, max_weight=4)
    assert abs(C.get((), 0) - 1.0) < 1e-12
    assert abs(C.get((1,), 0) - 3.0) < 1e-12
    assert abs(C.get((2,), 0) - 2.0) < 1e-12


def test_geode_convolution_weight_crop():
    # Ensure weight cropping removes high-degree terms
    A = {(): 1.0 + 0.0j, (1,): 1.0 + 0.0j}  # t2^0 + t2^1 (since axis 0 corresponds to t2)
    B = {(): 1.0 + 0.0j, (1,): 1.0 + 0.0j}
    # With max_weight=4, the product has term t2^2 allowed; with 1 it should be cropped
    C1 = geode_convolution_dict(A, B, num_vars=1, max_weight=4)
    assert (2,) in C1
    C2 = geode_convolution_dict(A, B, num_vars=1, max_weight=1)
    assert (2,) not in C2


