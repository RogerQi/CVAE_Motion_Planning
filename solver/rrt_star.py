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
    
    def update_parent(self, new_parent, new_cost):
        self.parent = new_parent
        assert self.ultimate_root == new_parent.ultimate_root
        self.local_cost = new_cost
    
    def __eq__(self, other):
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)

def bidirectional_rrt_star_base(start_conf, goal_conf, sampling_func, interpolate_func,
        metric_func, test_cfree_func, max_iter = 1000000, k = 5):
    '''
    Abstract bidirectional RRT* solver.

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
        for test_pt in eps_pts:
            if not test_cfree_func(test_pt):
                # No viable path...
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
        best_node = None
        path_feasible_lut = []
        for i, (state_i, node_i) in enumerate(k_nearest_nodes):
            path_feasible_bool = is_path_feasible(sampled_conf, state_i)
            path_feasible_lut.append(path_feasible_bool)
            if not path_feasible_bool:
                continue
            if best_node is None:
                best_node = node_i
            else:
                # Uncommenting these two lines enforces new node to adhere to nearest tree
                # but seems to make things slow...
                # if node_i.ultimate_root != best_node.ultimate_root:
                #     continue
                old_cost = best_node.local_cost + metric_func(best_node.state, sampled_conf)
                new_cost = node_i.local_cost + metric_func(state_i, sampled_conf)
                if new_cost < old_cost:
                    best_node = node_i
        if best_node is None:
            continue
        # Add this node to roadmap
        traversal_cost = metric_func(sampled_conf, best_node.state)
        new_node = node(sampled_conf, best_node, traversal_cost)
        nn_structure.add(new_node.state, new_node)
        # 2. See if it's possible to link from here to a node in different tree
        #       (RRT*): and also, test if newly sampled conf being the parent of K nearest node
        #                  can decrease cost of that node (has to happen in the same tree).
        for i, (state_i, node_i) in enumerate(k_nearest_nodes):
            if path_feasible_lut[i]:
                if node_i.ultimate_root != new_node.ultimate_root:
                    # Add an edge to the graph
                    rrt_graph.add_e(new_node, node_i)
                else:
                    # These two nodes are in the same tree
                    node_i_old_cost = node_i.local_cost
                    node_i_new_cost = metric_func(state_i, sampled_conf) + new_node.local_cost
                    if node_i_new_cost < node_i_old_cost:
                        node_i.update_parent(new_node, node_i_new_cost)
        # Is this solution or is it just fantasy?
        if rrt_graph.is_solved():
            # Found solution!
            return rrt_graph.get_soln()
    return None # No solution found in given iters!