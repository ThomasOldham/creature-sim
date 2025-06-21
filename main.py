from board import Board
from execution_timer import ExecutionTimer
from genome import Genome
from neural_network import NeuralNetwork
from simulation import Simulation
from IPython import embed
import action
import cell_stats
import creature_stats
import creature
import network_outputs
import numpy as np

np.seterr(all='raise')
np.seterr(under='ignore')
sim = Simulation(40, 20, random_seed=42)
board = sim.board
creature_storage = board.creature_storage
sim.run_round()

index = board.add_creature(Genome())
board.creatures[5,10] = index
creature_storage.grid_position[index] = [10, 5]
creature_storage.stats[index, creature.MASS] = 100.0
creature.recalculate_min_mass(creature_storage)

index = board.add_creature(Genome())
board.creatures[10,10] = index
creature_storage.grid_position[index] = [10, 10]
creature_storage.stats[index, creature.MASS] = 100.0
creature.recalculate_min_mass(creature_storage)

index = board.add_creature(Genome())
board.creatures[15,10] = index
creature_storage.grid_position[index] = [15, 10]
creature_storage.stats[index, creature.MASS] = 100.0
creature.recalculate_min_mass(creature_storage)

sim.run_round()
timer = ExecutionTimer()
def run_rounds(n: int):
    for _ in range(n):
        sim.run_round()
embed()