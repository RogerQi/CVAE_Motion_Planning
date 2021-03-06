import numpy as np
import numpy.linalg as npla

from structures.nearestneighbors import NearestNeighbors

class node(object):
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent

def rrt_base(start_conf, sampling_func, interpolate_func, metric_func, test_goal_func, test_cfree_func, max_iter = 1000000):
    '''
    Abstract naive RRT solver.

    Args:
        start_conf: a numpy array representing start configuration. Assumed collision-free.
        sampling_func: a function which samples a point in the configuration space.
            Sidenote: this can be non-uniform!
        metric_func: cost function that gives the edge weight between two configuration
        test_goal_func: pointer to a function which returns True if given configuration
            can be considered as goal; False otherwise.
        test_cfree_func: pointer to a function which tests if given configuration is collision-free.
    '''
    # TODO: use more efficient data structures
    root_node = node(start_conf, None)
    nn_structure = NearestNeighbors(metric_func, method = 'bruteforce')
    nn_structure.add(root_node.state, root_node)
    for n_iter in range(max_iter):
        sampled_conf = sampling_func()
        if not test_cfree_func(sampled_conf):
            continue
        nearest_node = nn_structure.nearest(sampled_conf)[1] # pt, data
        # Test if it's legal to travel from NN node to sampled node
        eps_pts = interpolate_func(nearest_node.state, sampled_conf)
        collision_flag = False
        for test_pt in eps_pts:
            if not test_cfree_func(test_pt):
                # No direct path...
                collision_flag = True
                break
        if collision_flag:
            continue
        # Test passed!
        # Add this node to roadmap
        new_node = node(sampled_conf, nearest_node)
        nn_structure.add(new_node.state, new_node)
        # Is this solution or is it just fantasy?
        if test_goal_func(new_node.state):
            # Found solution!
            ret = []
            cur_node = new_node
            while cur_node is not None:
                ret.append(cur_node.state)
                cur_node = cur_node.parent
            ret = ret[::-1]
            return ret
    return None # No solution found in given iters!