import numpy as np
import torch

from tarp.encoders.pooling import cls_pool, mean_pool


def test_mean_pool_ignores_padding():
    # 1 example, 3 tokens, hidden dim 2; third token is padding (mask 0).
    h = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    m = torch.tensor([[1, 1, 0]])
    out = mean_pool(h, m).numpy()
    assert np.allclose(out, [[2.0, 2.0]])  # mean of the first two only


def test_mean_pool_all_tokens():
    h = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    m = torch.tensor([[1, 1]])
    assert np.allclose(mean_pool(h, m).numpy(), [[2.0, 3.0]])


def test_cls_pool_takes_first_token():
    h = torch.tensor([[[5.0, 7.0], [0.0, 0.0]]])
    m = torch.tensor([[1, 1]])
    assert np.allclose(cls_pool(h, m).numpy(), [[5.0, 7.0]])
