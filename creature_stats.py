import numpy as np

# Private stats that linearly translate to input features
PRIVATE_LINEAR_FEATURES_START = 0
SUB_X = 0 # x-component of sub-position within grid cell
SUB_Y = 1 # y-component of sub-position within grid cell
LAST_DX = 2 # x-component of direction of last action
LAST_DY = 3 # y-component of direction of last action
LAST_SUCCESS = 4 # success of last action
PRIVATE_LINEAR_FEATURES_END = 5

# Private stats that need log10(1 + x) transformation
PRIVATE_LOG_FEATURES_START = 5
AGE = 5

# Upgradeable stats, a subset of private log features
UPGRADEABLE_STATS_START = 6
MAX_HP = 6
EAT_RATE = 7
MOVE_RATE = 8
ATTACK_POWER = 9
HEAL_POWER = 10
UPGRADEABLE_STATS_END = 11
PRIVATE_LOG_FEATURES_END = 11

# Public stats that need log10(1 + x) transformation
PUBLIC_LOG_FEATURES_START = 11
MASS = 11
PUBLIC_LOG_FEATURES_END = 12

# Private stats that need x/mass transformation
PRIVATE_MASS_FRACTION_FEATURES_START = 12
MIN_MASS = 12
LAST_COST = 13
PRIVATE_MASS_FRACTION_FEATURES_END = 14

# Private stats that need x/maxHP transformation
PRIVATE_MAX_HP_FRACTION_FEATURES_START = 14
LAST_DAMAGE_RECEIVED = 14
PRIVATE_MAX_HP_FRACTION_FEATURES_END = 15

# Public stats that need x/maxHP transformation
PUBLIC_MAX_HP_FRACTION_FEATURES_START = 15
DAMAGE = 15
PUBLIC_MAX_HP_FRACTION_FEATURES_END = 16

# Private stats that need x/lastDamage transformation
PRIVATE_LAST_DAMAGE_FRACTION_FEATURES_START = 16
LAST_DAMAGE_DX_SUM = 16
LAST_DAMAGE_DY_SUM = 17
PRIVATE_LAST_DAMAGE_FRACTION_FEATURES_END = 18

# Miscellaneous stats not directly used in features
LAST_ACTION = 18
BRAIN_MASS = 19 # cached at construction

COUNT = 20

NUM_PRIVATE_FEATURES = \
    PRIVATE_LINEAR_FEATURES_END - PRIVATE_LINEAR_FEATURES_START \
    + PRIVATE_LOG_FEATURES_END - PRIVATE_LOG_FEATURES_START \
    + PRIVATE_MASS_FRACTION_FEATURES_END - PRIVATE_MASS_FRACTION_FEATURES_START \
    + PRIVATE_MAX_HP_FRACTION_FEATURES_END - PRIVATE_MAX_HP_FRACTION_FEATURES_START \
    + PRIVATE_LAST_DAMAGE_FRACTION_FEATURES_END - PRIVATE_LAST_DAMAGE_FRACTION_FEATURES_START

NUM_PUBLIC_FEATURES = \
    PUBLIC_LOG_FEATURES_END - PUBLIC_LOG_FEATURES_START \
    + PUBLIC_MAX_HP_FRACTION_FEATURES_END - PUBLIC_MAX_HP_FRACTION_FEATURES_START
 
NUM_UPGRADEABLE_STATS = UPGRADEABLE_STATS_END - UPGRADEABLE_STATS_START

UPGRADEABLE_STAT_MASS_CONTRIBUTIONS = np.array([
    1.0,  # MAX_HP
    1.0,  # EAT_RATE
    1000.0,  # MOVE_RATE
    1.0,  # ATTACK_POWER
    10.0,  # HEAL_POWER
], dtype=np.float64)

UPGRADE_COSTS = np.array([
    10.0,  # MAX_HP
    10.0,  # EAT_RATE
    10000.0,  # MOVE_RATE
    10.0,  # ATTACK_POWER
    100.0,  # HEAL_POWER
], dtype=np.float64)

STARTING_VALUES = np.array([
    0.5,  # x sub-position
    0.5,  # y sub-position
    0.0,  # last dx
    0.0,  # last dy
    0.0,  # last success
    0.0,  # age
    10.0,  # max hp
    0.0,  # eat rate
    0.0,  # move rate
    0.0,  # attack power
    0.0,  # heal power
    np.nan,  # mass
    np.nan,  # min mass
    0.0,  # last cost
    0.0,  # last damage received
    0.0,  # damage
    0.0,  # last damage dx sum
    0.0,  # last damage dy sum
    -1.0,  # last action
    np.nan,  # brain mass
])