import random

import torch

from dcad.model import DCADActivityModel
from dcad.dcad import apply_mask
from dcad.em_dcad import EMDCADConfig, TransitionRarityMasker


def test_model_forward_shape():
    model = DCADActivityModel(6, 8, 2, 1, 16, 0.0, 4, 0)
    result = model(
        torch.tensor([[3, 1]]),
        torch.ones((1, 2), dtype=torch.long),
        torch.tensor([0.5]),
    )
    assert result.shape == (1, 2, 6)


def test_event_masking_core():
    masker = TransitionRarityMasker([[3, 4], [3, 5]], EMDCADConfig())
    positions = masker.sample_positions([3, 4], 0.5, random.Random(2026))
    inputs, labels, ratio = apply_mask([3, 4], 1, positions)
    assert len(inputs) == len(labels) == 2
    assert ratio == 0.5
