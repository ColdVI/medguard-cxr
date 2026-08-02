"""CheXpert uncertainty-policy ablation tests."""

import pytest
import torch

from medguard.data.chexpert import apply_uncertainty_policy


@pytest.mark.parametrize(
    ("policy", "expected", "mask"),
    [
        ("u_zero", [0.0, 0.0, 1.0], [True, True, True]),
        ("u_one", [1.0, 0.0, 1.0], [True, True, True]),
        ("u_ignore", [0.0, 0.0, 1.0], [False, True, True]),
    ],
)
def test_uncertainty_policies(policy: str, expected: list[float], mask: list[bool]) -> None:
    targets, loss_mask = apply_uncertainty_policy(torch.tensor([-1.0, 0.0, 1.0]), policy)

    assert targets.tolist() == expected
    assert loss_mask.tolist() == mask
