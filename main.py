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

sim = Simulation(40, 20, random_seed=42)
sim.run_round()
index = sim.board.add_creature(Genome())
sim.board.creature_storage.stats[index, creature.MASS] = 100.0
creature.recalculate_min_mass(sim.board.creature_storage)
sim.run_round()
embed()