import numpy as np
import numpy.linalg as npla

from structures.nearestneighbors import NearestNeighbors
from structures.rrt_graph import rrt_multi_root_graph

class node(object):
    node_id = 0
    def __init__(self, state, parent, cost_increment = 0):
        self.state = state
        self.parent = parent
        self.id = node.node_id
        node.node_id += 1
        if parent is not None:
            self.ultimate_root = parent.ultimate_root
            self.local_cost = parent.local_cost + cost_increment
        else:
            self.ultimate_root = self
            self.local_cost = 0
    
    def __eq__(self, other):
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)

def bidirectional_rrt_base(start_conf, goal_conf, sampling_func, interpolate_func,
        metric_func, test_cfree_func, max_iter = 1000000, k = 20):
    '''
    Abstract bidirectional RRT solver.

    Args:
        start_conf: a numpy array representing start configuration. Assumed collision-free.
        goal_conf: a numpy array representing goal configuration. Assumed collision-free.
        sampling_func: a function which samples a point in the configuration space.
            Sidenote: this can be non-uniform!
        metric_func: cost function that gives the edge weight between two configuration
        test_cfree_func: pointer to a function which tests if given configuration is collision-free.
        k: In each iteration, the nearest k nodes will be examined.
    '''
    # TODO: use more efficient data structures
    def is_path_feasible(conf_a, conf_b):
        eps_pts = interpolate_func(conf_a, conf_b)
        viable_flag = True
        for test_pt in eps_pts:
            if not test_cfree_func(test_pt):
                # No viable path...
                viable_flag = False
                break
        if viable_flag:
            return True
        second_eps_pts = interpolate_func(conf_a, conf_b, reverse = True)
        for test_pt in second_eps_pts:
            if not test_cfree_func(test_pt):
                return False
        return True

    root_node = node(start_conf, None)
    goal_node = node(goal_conf, None)
    nn_structure = NearestNeighbors(metric_func, method = 'bruteforce')
    nn_structure.add(root_node.state, root_node)
    nn_structure.add(goal_node.state, goal_node)
    rrt_graph = rrt_multi_root_graph(root_node, goal_node, metric_func)
    # Start iters
    for n_iter in range(max_iter):
        sampled_conf = sampling_func()
        if not test_cfree_func(sampled_conf):
            continue
        k_nearest_nodes = nn_structure.knearest(sampled_conf, k) # list of (state, node)
        # 1. See if we can connect sampled node to the nearest node
        nearest_node = k_nearest_nodes[0][1]
        if not is_path_feasible(sampled_conf, nearest_node.state):
            continue
        # Add this node to roadmap
        traversal_cost = metric_func(sampled_conf, nearest_node.state)
        new_node = node(sampled_conf, nearest_node, traversal_cost)
        nn_structure.add(new_node.state, new_node)
        # 2. See if it's possible to link from here to a node in different tree
        for n in k_nearest_nodes:
            n = n[1] # unpack to node
            if n.ultimate_root != new_node.ultimate_root:
                if is_path_feasible(new_node.state, n.state):
                    # Add an edge to the graph
                    rrt_graph.add_e(new_node, n)
        # Is this solution or is it just fantasy?
        if rrt_graph.is_solved():
            # Found solution!
            return rrt_graph.get_soln()
    return None # No solution found in given iters!