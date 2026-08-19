import numpy as np

from tarp.metrics.cka import cka_per_layer, linear_cka


def test_cka_self_is_one():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((150, 32))
    assert abs(linear_cka(X, X) - 1.0) < 1e-9


def test_cka_rotation_invariant():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((150, 32))
    Q, _ = np.linalg.qr(rng.standard_normal((32, 32)))
    assert abs(linear_cka(X, X @ Q) - 1.0) < 1e-6


def test_cka_low_for_unrelated():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((150, 32))
    Z = rng.standard_normal((150, 32))
    assert linear_cka(X, Z) < 0.5


def test_cka_per_layer_shapes():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((4, 100, 16))
    out = cka_per_layer(a, a)
    assert len(out) == 4 and all(abs(v - 1.0) < 1e-9 for v in out)
