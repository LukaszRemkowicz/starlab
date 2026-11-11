"""Fixed candidate search order shared by reference and compiled detection."""

PREFERRED_SEED_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (-1, 1),
    (-1, -1),
    (1, -1),
)
