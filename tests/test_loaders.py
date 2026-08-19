import numpy as np
import pytest

from tarp.data import load_task
from tarp.registry import resolve_dataset


@pytest.mark.parametrize(
    "key,K", [("trec", 6), ("banking77", 77), ("clinc150", 150), ("emotion", 6)]
)
def test_label_counts_and_encoding(key, K):
    td = load_task(resolve_dataset(key), seed=0)
    assert td.num_labels == K
    assert len(td.label_names) == K
    for texts, labels in (td.train, td.val, td.test):
        assert len(texts) == len(labels)
        assert labels.min() >= 0 and labels.max() < K
    # every class appears in train
    assert len(np.unique(td.train[1])) == K


def test_clinc_drops_oos():
    td = load_task(resolve_dataset("clinc150"), seed=0)
    assert "oos" not in td.label_names
    assert td.num_labels == 150


def test_sst2_uses_validation_for_eval():
    # SST-2 test labels are hidden; loader should evaluate on validation (872 rows).
    td = load_task(resolve_dataset("sst2"), seed=0)
    assert td.num_labels == 2
    assert len(td.test[0]) == 872
