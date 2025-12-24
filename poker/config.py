"""
Configuration constants for poker agent training and evaluation.

These values define the initial wealth distribution at the start of each training episode.
During episodes, wealth will vary (down to 0 when losing, higher when winning).
Sanity checks should use initial wealth near this range to ensure in-distribution testing.
"""

# Training configuration: initial wealth range at episode start
# Each episode begins with wealth randomly sampled from [MIN, MAX]
MIN_INITIAL_WEALTH = 15
MAX_INITIAL_WEALTH = 35
TYPICAL_INITIAL_WEALTH = (MIN_INITIAL_WEALTH + MAX_INITIAL_WEALTH) // 2  # 25
