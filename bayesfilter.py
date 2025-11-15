import numpy as np
import heapq

from pacman_module.game import Agent, Directions, manhattanDistance, Grid


# A constant for PacmanAgent's move options
MOVE_OPTIONS = [
    (0, 1, Directions.NORTH),
    (0, -1, Directions.SOUTH),
    (1, 0, Directions.EAST),
    (-1, 0, Directions.WEST)
]


class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()
        self.ghost = ghost

    def transition_matrix(self, walls: Grid, position):
        """Builds the transition matrix T_t = P(X_t | X_{t-1})
        given the current Pacman position.

        Arguments:
            walls: The W x H grid of walls.
            position: The current position of Pacman.

        Returns:
            The W x H x W x H transition matrix T_t.
            The element (i, j, k, l) of T_t is the probability
            P(X_t = (k, l) | X_{t-1} = (i, j)) for the ghost to move
            from (i, j) to (k, l).
        """
        w, h = walls.width, walls.height
        # Use deltas for N, S, E, W
        neighbour_deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # Creates the return value
        return_value = np.zeros((w, h, w, h))

        # Set ghost behavior based on type
        fear_factor = 0.0
        if self.ghost == "afraid":
            fear_factor = 1.0
        elif self.ghost == "terrified":
            fear_factor = 3.0

        # Pre-calculate weights
        away_weight = 2**fear_factor
        closer_weight = 1.0

        # For each possible start cell (i, j)
        for i in range(w):
            for j in range(h):
                # Skip wall cells as ghosts cannot be here
                if walls[i][j]:
                    continue

                start_pos = (i, j)
                start_dist_to_pac = manhattanDistance(position, start_pos)

                legal_neighbors = []
                weights = []

                # Check all 4 possible actions (N, S, E, W)
                for dx, dy in neighbour_deltas:
                    end_pos = (i + dx, j + dy)

                    # If the resulting cell (k, l) is not a wall...
                    if not walls[end_pos[0]][end_pos[1]]:
                        legal_neighbors.append(end_pos)

                        # Check distance from the *new* position
                        end_dist_to_pac = manhattanDistance(position, end_pos)

                        # Apply the weighting logic
                        if end_dist_to_pac >= start_dist_to_pac:
                            weights.append(away_weight)
                        else:
                            weights.append(closer_weight)

                # Normalize the weights to get probabilities
                total_weight = sum(weights)

                if total_weight > 0:
                    for (k, l), weight in zip(legal_neighbors, weights):
                        prob = weight / total_weight
                        return_value[i, j, k, l] = prob
                # If total_weight is 0, ghost is trapped and prob is 0

        return return_value

    def observation_matrix(self, walls: Grid, evidence, position):
        """Builds the observation matrix O_t = P(e_t | X_t)
        given a noisy ghost distance evidence e_t and
        the current Pacman position.

        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The W x H observation matrix O_t.
        """
        w, h = walls.width, walls.height
        return_value = np.zeros((w, h))

        # Pre-calculated probabilities for Binom(n=4, p=0.5)
        # The noise is z - np = z - 2
        # P(noise) = P(z = noise + 2)
        noise_probs = {
            -2: 1/16.0,  # P(z=0)
            -1: 4/16.0,  # P(z=1)
            0: 6/16.0,   # P(z=2)
            1: 4/16.0,   # P(z=3)
            2: 1/16.0    # P(z=4)
        }

        for i in range(w):
            for j in range(h):
                # P(evidence | X_t = (i,j))

                # If (i,j) is a wall, prob is 0
                if walls[i][j]:
                    return_value[i, j] = 0.0  # Explicitly set to 0
                    continue

                # Get true distance (dm) from Pacman to this cell
                dm = manhattanDistance(position, (i, j))

                # Calculate the noise value
                # e = dm + noise_val => noise_val = e - dm
                noise = evidence - dm

                # Get the probability of this noise value from
                # the binomial distribution.
                # If noise is not in [-2, 2], prob is 0.
                prob = noise_probs.get(noise, 0.0)

                return_value[i, j] = prob

        return return_value

    def update(self, walls: Grid, belief, evidence, position):
        """Updates the previous ghost belief state

            b_{t-1} = P(X_{t-1} | e_{1:t-1})

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            belief: The belief state for the previous ghost position b_{t-1}.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The updated ghost belief state b_t as a W x H matrix.
        """

        transition = self.transition_matrix(walls, position)
        observation = self.observation_matrix(walls, evidence, position)

        w, h = walls.width, walls.height

        time_updated_belief = np.zeros((w, h))
        for ip in range(w):  # Target cell (k, l) in formula
            for jp in range(h):
                if walls[ip][jp]:
                    continue

                # Sum over all previous states (i, j)
                for i in range(w):
                    for j in range(h):
                        if walls[i][j]:
                            continue
                        time_updated_belief[ip, jp] += (
                            transition[i, j, ip, jp] * belief[i, j]
                        )

        # Apply observation
        new_belief = observation * time_updated_belief

        # Normalize
        s = np.sum(new_belief)
        if s > 10e-9:
            new_belief /= s
        '''else:
            # If belief became all zeros.
            # Return a uniform belief.
            num_free_cells = w * h - np.sum(walls.data)
            if num_free_cells > 0:
                prob = 1.0 / num_free_cells
                for i in range(w):
                    for j in range(h):
                        if not walls[i][j]:
                            new_belief[i, j] = prob'''

        return new_belief

    def get_action(self, state):
        """Updates the previous belief states given the current state.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            The list of updated belief states.
        """

        walls = state.getWalls()
        beliefs = state.getGhostBeliefStates()
        eaten = state.getGhostEaten()
        evidences = state.getGhostNoisyDistances()
        position = state.getPacmanPosition()

        new_beliefs = [None] * len(beliefs)

        for i in range(len(beliefs)):
            if eaten[i]:
                new_beliefs[i] = np.zeros_like(beliefs[i])
            else:
                new_beliefs[i] = self.update(
                    walls,
                    beliefs[i],
                    evidences[i],
                    position,
                )

        return new_beliefs


class PacmanAgent(Agent):
    """Pacman agent that tries to eat ghosts given belief states."""

    def __init__(self):
        super().__init__()

    def _run_astar(self, start_pos, end_pos, walls):
        """
        Runs A* search to find the shortest path from start_pos to end_pos.

        Returns:
            A tuple (cost, path) or (inf, []) if no path is found.
        """
        # Priority Queue stores: (priority, g_cost, path_list, current_pos)
        # priority = g_cost + h_cost
        pq = []

        g_cost = 0
        h_cost = manhattanDistance(start_pos, end_pos)
        priority = g_cost + h_cost

        # g_cost is in the tuple for tie-breaking (favors shorter paths)
        heapq.heappush(pq, (priority, g_cost, [], start_pos))

        visited = {start_pos}  # Use a set for visited nodes

        while pq:
            _priority, g_cost, path, current_pos = heapq.heappop(pq)

            # If target is found, return the path and its cost
            if current_pos == end_pos:
                return g_cost, path

            px, py = current_pos

            for dx, dy, action in MOVE_OPTIONS:
                next_pos = (px + dx, py + dy)

                if (not walls[next_pos[0]][next_pos[1]]) and \
                        (next_pos not in visited):

                    visited.add(next_pos)  # Mark as visited

                    new_g_cost = g_cost + 1
                    new_h_cost = manhattanDistance(next_pos, end_pos)
                    new_priority = new_g_cost + new_h_cost

                    new_path = path + [action]
                    heapq.heappush(
                        pq, (new_priority, new_g_cost, new_path, next_pos)
                    )

        # If the loop finishes without finding the target
        return float('inf'), []

    def _get_action(self, walls, beliefs, eaten, position):
        """
        Arguments:
            walls: The W x H grid of walls.
            beliefs: The list of current ghost belief states.
            eaten: A list of booleans indicating which ghosts have been eaten.
            position: The current position of Pacman.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        min_path_cost = float('inf')
        best_first_move = Directions.STOP

        # Find the shortest path to the *closest* uneaten ghost
        for i in range(len(beliefs)):
            # If ghost is eaten or belief is all zeros, skip it
            if eaten[i] or np.sum(beliefs[i]) == 0:
                continue

            belief = beliefs[i]
            # Find the (x, y) coords of the max probability in the belief grid
            ghost_pos_flat_index = np.argmax(belief)
            ghost_pos = np.unravel_index(ghost_pos_flat_index, belief.shape)

            # If we are already on the target, stop and eat
            if position == ghost_pos:
                return Directions.STOP

            # Run A* to find the shortest path to this ghost
            path_cost, path = self._run_astar(position, ghost_pos, walls)

            # Check if this ghost is the new closest target
            if path and path_cost < min_path_cost:
                min_path_cost = path_cost
                best_first_move = path[0]  # Get the first move

        # Return the first step of the best path found
        return best_first_move

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        return self._get_action(
            state.getWalls(),
            state.getGhostBeliefStates(),
            state.getGhostEaten(),
            state.getPacmanPosition(),
        )
