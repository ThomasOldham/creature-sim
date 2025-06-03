from typing import Optional
from board import Board
import numpy as np

class Simulation:
    def __init__(self, width: int, height: int, random_seed: Optional[int] = None):
        if random_seed:
            np.random.seed(random_seed)
        self.board = Board(width, height)

    def run_round(self) -> None:
        self.board.setup_round()