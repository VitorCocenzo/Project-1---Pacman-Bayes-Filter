# Instructions

In this project, Pacman got tired of ghosts wandering around him, so he
bought a magic gun that makes the ghosts edible. But while he shot them,
he figured out that the gun also made them invisible. Fortunately, he
also got his hands on a rusty distance sensor.\
The sensor returns a **noisy Manhattan distance** between Pacman and
each ghost, which Pacman can use as evidence to find the ghost
positions.

The noisy distance, denoted ( e ), results from the addition of noise to
the true Manhattan distance. The noise is sampled from a **binomial
distribution centered around 0**:

``` math
e = \text{ManhattanDistance}(\text{Pacman}, \text{Ghost}) + z - np
```

where

``` math
z \sim \text{Binom}(n, p), \quad n = 4,\quad p = 0.5.
```

Pacman knows that the ghosts are afraid of him and are more likely to
take actions that move them away from him.\
Their exact action policies (afraid, fearless, terrified) should be
deduced from the `ghostAgents.py` file.

Your task is to design an intelligent agent based on the **Bayes filter
algorithm** for locating and eating all the ghosts in the maze.

------------------------------------------------------------------------

## 1. Bayes Filter Implementation

Implement the Bayes filter algorithm to compute Pacman's belief state of
the ghost positions.

To do so, fill in the three methods of the `BeliefStateAgent` class:

-   `transition_matrix`
-   `observation_matrix`
-   `update`

The `update` method **must rely on** both `transition_matrix` and
`observation_matrix`.

------------------------------------------------------------------------

## 2. Pacman Agent Implementation

Implement a Pacman agent whose goal is to eat all the ghosts as fast as
possible.

At each step, the agent has access only to:

-   its **own position**
-   the **current belief state** of the ghost positions

To do so, fill in the `_get_action` method of the `PacmanAgent` class.

------------------------------------------------------------------------

## Running the project

First, download and extract the archive of the project in a directory of
your choice.

Use the following command to run your Bayes filter implementation
against **a single afraid ghost** in the **large_filter** layout:

``` bash
python run.py --ghost afraid --nghosts 1 --layout large_filter --seed 42
```

When several ghosts are present in the maze, they all run the same
policy (e.g., all *afraid*).\
The random seed of the game can be changed with the `--seed` option.
