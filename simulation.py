from typing import Optional, Tuple
import action
from board import Board
import numpy as np
from action import ACTION_FUNCTIONS, attack_action
from execution_timer import timer_decorator

import creature
import creature_stats
import network_outputs

class Simulation:
    def __init__(self, width: int, height: int, random_seed: Optional[int] = None):
        if random_seed:
            np.random.seed(random_seed)
        self.board = Board(width, height)

    @timer_decorator('Simulation.run_round')
    def run_round(self) -> None:
        self.board.setup_round()
        self._creature_turns()
        self.board.wrapup_round()
    
    @timer_decorator('Simulation._creature_turns')
    def _creature_turns(self) -> None:
        action_kinds, action_params = self._decide_actions()
        creature.reset_short_term_memory(self.board.creature_storage)
        action_results = self._execute_actions(action_kinds, action_params)
        self._apply_action_results(action_results)

    @timer_decorator('Simulation._decide_actions')
    def _decide_actions(self) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self._get_outputs()
        action_kinds = creature.decide_action_kind(outputs)
        action_params = creature.action_params(outputs, self.board.creature_storage)
        return action_kinds, action_params
    
    @timer_decorator('Simulation._get_outputs')
    def _get_outputs(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        if not out:
            out = np.empty((self.board.creature_storage.used_row_count(), network_outputs.COUNT), dtype=np.float64)
        inputs = self.board.all_features()
        creature_storage = self.board.creature_storage
        for index in range(creature_storage.used_row_count()):
            if not creature_storage.is_alive[index]:
                continue
            vision_radius = creature_storage.vision_radius[index]
            features_index = creature_storage.features_index[index]
            out[index] = creature_storage.network[index].forward(inputs[vision_radius][features_index])
        return out
    
    @timer_decorator('Simulation._execute_actions')
    def _execute_actions(self, action_kinds: np.ndarray, action_params: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        if not out:
            out = np.empty((len(action_kinds), action.RESULT_SIZE), dtype=np.float64)

        # attack actions
        attack_mask = self.board.creature_storage.is_alive & (action_kinds == 0)
        if attack_mask.any():
            attack_action(attack_mask, action_params, self.board.creature_storage, self.board, out=out)

        # the other actions
        action_kind_masks = [
            self.board.creature_storage.is_alive & (action_kinds == kind)
            for kind in range(1, network_outputs.ACTION_KINDS_COUNT)
        ]
        for kind, mask in enumerate(action_kind_masks):
            if not mask.any():
                continue
            ACTION_FUNCTIONS[kind](mask, action_params, self.board.creature_storage, self.board, out)
        return out
    
    @timer_decorator('Simulation._apply_action_results')
    def _apply_action_results(self, action_results: np.ndarray) -> None:
        creature_storage = self.board.creature_storage
        stats = creature_storage.stats
        np.subtract(stats[:, creature_stats.MASS], action_results[:, action.RESULT_COST], out=stats[:, creature_stats.MASS])
        stats[:, creature_stats.LAST_SUCCESS] = action_results[:, action.RESULT_SUCCESS]
        stats[:, creature_stats.LAST_COST] = action_results[:, action.RESULT_COST]
        stats[:, creature_stats.LAST_ACTION] = action_results[:, action.RESULT_KIND]
        stats[:, creature_stats.LAST_DX] = action_results[:, action.RESULT_DIR_X]
        stats[:, creature_stats.LAST_DY] = action_results[:, action.RESULT_DIR_Y]