from random_maze1_3 import RandomMaze
import numpy as np
import gymnasium as gym
import networkx as nx

'''
TODO:
    - Experiment setup so code can be repeatedly called

Done:
    1. Max-flow based solution
    2. Shortest path based solution
'''

class GraphSolution:
    def __init__(self, env=None):
        r"""Graph-based solution to random maze problem. Implements the following algorithms:
        1. Shortest Path.
        2. Max-flow.
        """
        self.env = env
        self.graph = None
        self.adj_list = None

    def _linear_index(self, position):
        r"""Returns the linear index of ``position`` for ``m``x``n`` maze."""
        return position[0] * self.env.get_wrapper_attr('maze').shape[0] + position[1]

    def _grid_index(self, index):
        r"""Returns the grid index of ``position`` for ``m``x``n`` maze."""
        return [index // self.env.get_wrapper_attr('maze').shape[0], index % self.env.get_wrapper_attr('maze').shape[1]]

    def _action(self, src, dst):
        r"""Returns the action taken to go from ``src`` to ``dst``.
        
        Args:
            src (int): Source as linear index.
            dst (int): Destination as linear index.

        Returns:
            int: Action taken to go from ``src`` to ``dst``.
        """
        dst_pos = self._grid_index(dst)
        src_pos = self._grid_index(src)
        action = [dst_pos[0] - src_pos[0], dst_pos[1] - src_pos[1]]
        return self.env.get_wrapper_attr('motions').index(action)

    def create_adj_list(self):
        r"""Constructs a single source, single sink directional graph (S4DG) from the maze in form of adjacency list, stored in `self.adj_list`.       

        This construction ignores obstacles as vertices. It does not allow outgoing edges from the goal and incoming edges to the start.
        
        Additionally, an illegal move is not represented as an edge. This is because the agent will never be able to take that action.

        Edge weights are computed as the reward obtained by taking the corresponding action.

        Returns:
            None
        """
        adj_list = {}
        maze = self.env.get_wrapper_attr('_to_value')()
        motions = self.env.get_wrapper_attr('motions')

        # Traverse the positions
        for i,j in np.ndindex(maze.shape):
            # If the position is an obstacle or the goal, skip
            if maze[i,j] == 1 or maze[i,j] == 3:
                continue

            # For each legal action, find the dst vertex and compute weight
            for idx in range(len(motions)):
                action = motions[idx]
                new_position = [i + action[0], j + action[1]]

                # Skip if destination is start
                if maze[new_position[0], new_position[1]] == 2:
                    continue
                
                # Illegal Action
                if (new_position[0] < 0 or new_position[0] >= maze.shape[0]) or \
                    (new_position[1] < 0 or new_position[1] >= maze.shape[1]):
                    continue
                
                # Compute reward and add to adj list, reset the agent's position
                self.env.get_wrapper_attr('_objects').agent.positions = [[i,j]]
                _, reward, terminated, _, _ = self.env.get_wrapper_attr('step')(idx)
                self.env.get_wrapper_attr('_objects').agent.positions = self.env.get_wrapper_attr('_start_idx')

                # Better than hardcoding the failure's reward, check if terminated and negative reward
                # Which means that the agent hit an obstacle
                if terminated and reward < 0:
                    continue
                else:
                    # Test: See if the action is the same as the one computed by the graph
                    # reconst_action = self._action(self._linear_index((i,j)), self._linear_index(new_position))
                    adj_list[self._linear_index((i,j)), self._linear_index(new_position)] = reward

        self.adj_list = adj_list 

    def max_flow_solution(self):
        r"""Computes the max flow in the graph using Edmonds-Karp algorithm.

        Note that a new ``nx.DiGraph()`` is constructed from `self.adj_list` every time this method is called.

        Invoke this method after calling `create_graph_dense()`.

        Args:
            None.

        Returns:
            ``actions` (`list`): List of actions taken to achieve the max flow, starting from the source.
        """
        assert self.adj_list is not None, "Graph is not initialized. Call create_graph_dense() first."
        
        # Compute integer-valued graph from self.adj_list
        graph = nx.DiGraph()
        for (s,d), weight in self.adj_list.items():
            graph.add_edge(str(s),str(d), capacity=1)
        src, dst = str(self._linear_index(self.env.get_wrapper_attr('_start_idx')[0])), str(self._linear_index(self.env.get_wrapper_attr('_goal_idx')[0]))
        _, flow_dict = nx.algorithms.flow.maximum_flow(graph, src, dst, flow_func=nx.algorithms.flow.edmonds_karp)

        # Compute actions from flow graph
        actions, tmp = [], src
        while tmp != dst:
            flag = False
            for node in flow_dict[tmp]:
                if flow_dict[tmp][node] > 0:
                    actions.append(self._action(int(tmp), int(node)))
                    tmp = node
                    flag = True
                    break
            if flag == False:
                raise ValueError("No path found from source to destination.")
        return actions
    
    def shortest_path_solution(self):
        r"""Computes the shortest path between source and destination in the graph using Bellman-Ford algorithm.
        
        Bellman-Ford alternative is used as the graph is cyclic.
        
        Invoke this method after calling `create_graph_dense()`.
        
        Args:
            None.
            
        Returns:
            ``actions` (`list`): List of actions taken to achieve the shortest path, starting from the source.
        """
        assert self.adj_list is not None, "Graph is not initialized. Call create_graph_dense() first."
        
        # Compute integer-valued graph from self.adj_list
        graph = nx.DiGraph()
        for (s,d), weight in self.adj_list.items():
            graph.add_edge(str(s),str(d))
        src, dst = str(self._linear_index(self.env.get_wrapper_attr('_start_idx')[0])), str(self._linear_index(self.env.get_wrapper_attr('_goal_idx')[0]))
        path = nx.shortest_path(graph, source=src, target=dst, method='bellman-ford')

        # Compute actions from the path
        actions = []
        for i in range(len(path)-1):
            node = path[i]
            next_node = path[i+1]
            actions.append(self._action(int(node), int(next_node)))
        return actions

if __name__ == "__main__":
    import time
    from pathlib import Path
    env = RandomMaze(render_mode='human')
    for path in Path('./mazes_25').glob('*.pkl'):
        path = str(path.resolve())
        env.generate_maze(path)
        env.reset()
        t1 = time.time()
        maze_solver = GraphSolution(env)
        maze_solver.create_adj_list()
        t2 = time.time()
        actions = maze_solver.max_flow_solution()
        state = env.reset()[0]
        cum_reward = 0

        for action in actions:
            env.render()
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            cum_reward+=reward

        env.render()
        t3 = time.time()
        print(path)
        print("Cumulative reward: {}".format(cum_reward))
        print("Time taken to find shortest path: {:.3f}s".format(t3-t2))
        print("Time taken to find shortest path, including graph construction: {:.3f}s\n".format(t3-t1))
        input("Press Enter to continue...")    
    env.close()