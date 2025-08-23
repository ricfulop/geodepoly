import os


def test_numba_toggle_env_import():
    # Ensure importing util with toggle does not crash
    os.environ["GEODEPOLY_USE_NUMBA"] = "0"
    from importlib import reload
    import geodepoly.util as util

    reload(util)
    assert hasattr(util, "poly_eval")



