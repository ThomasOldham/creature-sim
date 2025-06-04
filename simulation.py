from typing import Optional, Tuple
from board import Board
import numpy as np

import creature
import network_outputs

class Simulation:
    def __init__(self, width: int, height: int, random_seed: Optional[int] = None):
        if random_seed:
            np.random.seed(random_seed)
        self.board = Board(width, height)

    def run_round(self) -> None:
        self.board.setup_round()
        self._creature_turns()
    
    def _creature_turns(self) -> None:
        action_kinds, action_params = self._creature_decisions()
        action_kind_masks = [
            self.board.creature_storage.is_alive & (action_kinds == kind)
            for kind in range(network_outputs.ACTION_KINDS_COUNT)
        ]

    def _creature_decisions(self) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self._get_outputs()
        action_kinds = creature.decide_action_kind(outputs)
        action_params = creature.action_params(outputs, self.board.creature_storage)
        return action_kinds, action_params
    
    def _get_outputs(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        if not out:
            out = np.empty((self.board.creature_storage.used_row_count(), network_outputs.COUNT), dtype=np.float64)
        inputs = self.board.all_features()
        creature_storage = self.board.creature_storage
        for index in range(creature_storage.used_row_count()):
            if not creature_storage.is_alive[index]:
                continue
            creature_storage.network[index].forward(inputs[index], out[index])
        return out