from dataclasses import dataclass, field
from typing import Tuple

# Points between two corners on each side of the measuring frame.
LEAK_SIDE_POINTS = 3


@dataclass(frozen=True)
class LeakConfig:
    """Light-leak removal, measured on the clear film around the frame."""

    leak_apply: bool = False
    # The measuring frame as a closed loop in raw-image normalized coords, clockwise from the top-left
    # corner: TL, top points, TR, right points, BR, bottom points, BL, left points.
    leak_points: Tuple[Tuple[float, float], ...] = field(default_factory=tuple)
    # 1.0 takes out the measured leak; under 1 leaves some, over 1 takes out more.
    leak_strength: float = 1.0

    def __post_init__(self) -> None:
        # JSON gives back lists; a frozen config must hold tuples to stay hashable.
        object.__setattr__(self, "leak_points", tuple((float(p[0]), float(p[1])) for p in self.leak_points))
        object.__setattr__(self, "leak_strength", float(self.leak_strength))
