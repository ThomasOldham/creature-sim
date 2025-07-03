A work-in-progress simulation in which neural-network-controlled agents ("creatures") compete to survive and reproduce. My goal is to see interesting adaptations evolve as creatures mutate over generations.

Portions of the codebase are AI-generated.

# Simulation overview

The simulation takes place over rounds on a square-grid board. Each round, each creature observes its surroundings and its own state. It passes that sense data to a neural network, whose output determines the creature's actions for the turn. Actions include moving, eating food, reproducing, and attacking other creatures.

At the beginning of each round, food is distributed around the board, and there is a small chance of simple new creatures being spontaneously created. Creatures age slightly and passively consume some stored food each round.

# Reproduction, mutation, and growth

When a creature reproduces, its neural network "brain" is inherited by the offspring with some random mutations. Some other immutable features are also inherited. For now, mutation with natural selection is the only way for adaptation to occur, so there is no actual machine learning.

Each creature always starts with the most basal possible outward, mutable characteristics (regardless of what its parent looked like), and can develop only by deciding to do so on its turn.

# Planned features and further steps

## Additional action types for creatures to take

So far I've added actions that are either necessary for MVP, or that I thought would be easy to implement. I plan to eventually add at least a few more, including a way to synthesize food, a way to intentionally share food, and dedicated communication methods.

## Persistent memory

Creatures should each have some bytes they can arbitrarily read and write from to help inform their actions.

## More sophisticated perception of other creatures

Creatures can currently tell very little about each other from looking. I want to add enough observable data points that creatures could plausibly learn to for example identify kin from non-kin, or identify probable predators and appealing prey.

## A simple GUI

At a minimum, the user should be able to save, load, observe, and step forward in the simulation.