from dataclasses import dataclass, field
from typing import Tuple

# Points between two corners on each side of the measuring frame.
LEAK_SIDE_POINTS = 3


@dataclass(frozen=True)
class LeakConfig:
    """Light-leak removal, measured on the clear film around the frame."""

    leak_apply: bool = False
    # The measuring frame as a closed loop in the straightened image's normalized coords (the crop's
    # space), clockwise from the top-left corner: TL, top points, TR, right points, BR, bottom points,
    # BL, left points.
    leak_frame: Tuple[Tuple[float, float], ...] = field(default_factory=tuple)
    # The crop rect the frame was placed against, or empty. A frame with a crop of its own maps the
    # loop from this rect onto its crop, so the frame follows each frame's crop across a roll.
    leak_anchor: Tuple[float, ...] = field(default_factory=tuple)
    # 1.0 takes out the measured leak; under 1 leaves some, over 1 takes out more.
    leak_strength: float = 1.0

    def __post_init__(self) -> None:
        # JSON gives back lists; a frozen config must hold tuples to stay hashable.
        object.__setattr__(self, "leak_frame", tuple((float(p[0]), float(p[1])) for p in self.leak_frame))
        anchor = tuple(float(v) for v in self.leak_anchor)
        object.__setattr__(self, "leak_anchor", anchor if len(anchor) == 4 else ())
        object.__setattr__(self, "leak_strength", float(self.leak_strength))
