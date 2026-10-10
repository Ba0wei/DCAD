"""Internal masking seeds for DCAD-family training scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DCADInternalSeeds:
    mask: int = 123
    validation_mask: int = 2026


DCAD_SEEDS = DCADInternalSeeds()


def dcad_seed_payload() -> dict[str, int]:
    return asdict(DCAD_SEEDS)
