import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import apk, mapk  # noqa: E402


def test_perfect():
    assert apk([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == 1.0


def test_no_hits():
    assert apk([1, 2], [3, 4, 5, 6, 7]) == 0.0


def test_empty_truth():
    assert apk([], [1, 2, 3]) == 0.0


def test_single_truth_at_rank3():
    # P(3)*1 / min(1,5) = 1/3
    assert apk([7], [1, 2, 7, 3, 4]) == pytest.approx(1 / 3)


def test_two_truth_hits_rank1_and_3():
    # (1/1 + 2/3) / min(2,5)
    assert apk([1, 2], [1, 9, 2, 8, 7]) == pytest.approx((1 + 2 / 3) / 2)


def test_truth_longer_than_k_and_cut():
    # 6 truths, denominator min(6,5)=5; hits at ranks 2 and 5: (1/2 + 2/5)/5 ; rank-6 prediction ignored
    assert apk([1, 2, 3, 4, 5, 6], [9, 1, 8, 7, 2, 3]) == pytest.approx((0.5 + 0.4) / 5)


def test_duplicates_not_recredited():
    assert apk([1], [1, 1, 1, 1, 1]) == 1.0
    assert apk([1, 2], [1, 1, 2]) == pytest.approx((1 + 2 / 3) / 2)


def test_mapk_mean_and_missing_user():
    actual = {"a": [1], "b": [2]}
    assert mapk(actual, {"a": [1, 3]}) == pytest.approx(0.5)
