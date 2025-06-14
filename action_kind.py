from typing import TypeAlias

# Action kinds as integer constants
ActionKind: TypeAlias = int
ACTION_NOTHING: ActionKind = -1  # placeholder action not selectable by neural network
ACTION_ATTACK: ActionKind = 0
ACTION_EAT: ActionKind = 1
ACTION_HEAL: ActionKind = 2
ACTION_UPGRADE: ActionKind = 3
ACTION_MOVE: ActionKind = 4
ACTION_REPRODUCE: ActionKind = 5
ACTION_KINDS_COUNT = 6 