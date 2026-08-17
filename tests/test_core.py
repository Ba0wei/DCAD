import random

import torch

from dcad.model import DCADActivityModel
from dcad.dcad import apply_mask
from dcad.em_dcad import EMDCADConfig, TransitionRarityMasker
from dcad.tn_dcad import TNDCADConfig, expand_trace_views, trace_contribution_weights


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


def test_tn_dcad_expands_all_views_with_beta_weights():
    config = TNDCADConfig(trace_weight_beta=0.5)
    levels = ["low", "medium", "high"]
    source_indices, mask_ratios, weights = expand_trace_views(
        levels, config, random.Random(2026)
    )

    assert source_indices == [0, 1, 1, 2, 2, 2]
    assert len(mask_ratios) == 6
    assert weights == [
        1.0,
        2 ** -0.5,
        2 ** -0.5,
        3 ** -0.5,
        3 ** -0.5,
        3 ** -0.5,
    ]


def test_tn_dcad_beta_controls_aggregate_trace_contribution():
    levels = ["low", "medium", "high"]
    for beta in (0.0, 0.5, 1.0, 2.0):
        per_view = trace_contribution_weights(levels, beta)
        aggregate = [weight * views for weight, views in zip(per_view, (1, 2, 3))]
        expected = [views**beta for views in (1, 2, 3)]
        assert aggregate == expected
