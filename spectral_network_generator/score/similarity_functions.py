from typing import List, Tuple
import numba
import numpy as np


@numba.njit(fastmath=True)
def score_best_matches_with_index(matching_pairs: np.ndarray, spec1: np.ndarray,
                                  spec2: np.ndarray, mz_power: float = 0.0,
                                  intensity_power: float = 1.0) -> Tuple[float, int, List[int], List[int]]:
    """Calculate cosine-like score by multiplying matches. Does require a sorted
    list of matching peaks (sorted by intensity product)."""
    score = float(0.0)
    used_matches = int(0)
    used1 = list()
    used2 = list()
    for i in range(matching_pairs.shape[0]):
        if not matching_pairs[i, 0] in used1 and not matching_pairs[i, 1] in used2:
            score += matching_pairs[i, 2]
            used1.append(matching_pairs[i, 0])  # Every peak can only be paired once
            used2.append(matching_pairs[i, 1])  # Every peak can only be paired once
            used_matches += 1

    # Normalize score:
    spec1_power = spec1[:, 0] ** mz_power * spec1[:, 1] ** intensity_power
    spec2_power = spec2[:, 0] ** mz_power * spec2[:, 1] ** intensity_power

    score = score/(np.sum(spec1_power ** 2) ** 0.5 * np.sum(spec2_power ** 2) ** 0.5)
    return score, used_matches, used1, used2
