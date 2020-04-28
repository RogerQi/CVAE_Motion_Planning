import heapq
import numpy as np
import numpy.linalg as npla
import time

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
            self.cost = parent.cost + cost_increment
        else:
            self.ultimate_root = self
            self.cost = 0
    
    def update_parent(self, new_parent, new_cost):
        self.parent = new_parent
        assert self.ultimate_root == new_parent.ultimate_root
        self.cost = new_cost

    def __lt__(self, other):
        return self.cost < other.cost
    
    def __gt__(self, other):
        return self.cost > other.cost
    
    def __eq__(self, other):
        return self.cost == other.cost
    
    def __ge__(self, other):
        return self.cost >= other.cost
    
    def __le__(self, other):
        return self.cost <= other.cost

    def __hash__(self):
        return hash(self.id)

def fmt_star_base(start_conf, goal_conf, test_goal_func, sampling_func, interpolate_func,
        metric_func, test_cfree_func, max_iter = 3000, k = 20):
    '''
    Abstract FMT* solver.

    Args:
        start_conf: a numpy array representing start configuration. Assumed collision-free.
        goal_conf: a numpy array representing goal configuration. Assumed collision-free.
        sampling_func: a function which samples a point in the configuration space.
            Sidenote: this can be non-uniform!
        metric_func: cost function that gives the edge weight between two configuration
        test_cfree_func: pointer to a function which tests if given configuration is collision-free.
        k: In each iteration, the nearest k nodes will be examined.
    '''
    implicit_v_unvisited = False
    assert implicit_v_unvisited == False, "Not implemented yet"
    def is_path_feasible(conf_a, conf_b):
        eps_pts = interpolate_func(conf_a, conf_b)
        for test_pt in eps_pts:
            if not test_cfree_func(test_pt):
                # No viable path...
                return False
        return True
    
    assert start_conf.shape == goal_conf.shape

    root_node = node(start_conf, None)

    if implicit_v_unvisited:
        pass
    else:
        V_unvisited = []
        while len(V_unvisited) < max_iter:
            new_conf = sampling_func(size = start_conf.shape)
            if test_cfree_func(new_conf):
                V_unvisited.append(new_conf)
        V_unvisited.append(goal_conf)
    V_open = [root_node]
    V_closed = []

    def test_neighbor(conf_a, conf_b):
        if metric_func(conf_a, conf_b) < 0.03:
            return True
        else:
            return False

    while len(V_open) > 0:
        print("V open len: {}".format(len(V_open)))
        print("V unvisited len: {}".format(len(V_unvisited)))
        cur_node = V_open[0]
        # Test goal
        if test_goal_func(cur_node.state):
            # Found solution
            ret = []
            while cur_node is not None:
                ret.append(cur_node.state)
                cur_node = cur_node.parent
            ret = ret[::-1]
            return ret
        if implicit_v_unvisited:
            assert False
        else:
            z_neighbors_x = []
            # It'd be best to reverse the order so that deletion is easier
            start_cp = time.time()
            for i in range(len(V_unvisited) - 1, -1, -1):
                if test_neighbor(cur_node.state, V_unvisited[i]):
                    z_neighbors_x.append(i)
            my_time = time.time() - start_cp
            print("Sort V unvisited time: {}".format(my_time))
        if not z_neighbors_x:
            heapq.heappop(V_open)
            continue # no neighbor
        print(cur_node.state)
        print("Z neighbor x len: {}".format(len(z_neighbors_x)))
        # Construct a smaller subset of V_open to speed up computation
        potential_V_open_candidates = sorted(V_open, key = lambda n : metric_func(n.state, cur_node.state))
        potential_V_open_candidates = potential_V_open_candidates[:1000]
        # cur_node is removed from V_open to V_closed
        heapq.heappop(V_open)
        for x_conf_idx in z_neighbors_x:
            # Find neighbor nodes to x_conf in V_open
            # and construct locally-optimal one-step connection from y to x
            best_x_node = None
            for y_node in potential_V_open_candidates:
                if not test_neighbor(V_unvisited[x_conf_idx], y_node.state):
                    continue
                # X and Y are neighbor!
                # LAZY: no collision check here.
                x_y_cost = metric_func(V_unvisited[x_conf_idx], y_node.state)
                if best_x_node is None:
                    best_x_node = node(V_unvisited[x_conf_idx], y_node, x_y_cost)
                else:
                    new_total_cost = x_y_cost + y_node.cost
                    if new_total_cost < best_x_node.cost:
                        best_x_node.update_parent(y_node, new_total_cost)
            if best_x_node is None:
                continue
            # Test path
            if not is_path_feasible(best_x_node.state, best_x_node.parent.state):
                continue
            # Viable path! Preserve best_x_node
            heapq.heappush(V_open, best_x_node)
            # Remove it from V_unvisited
            del(V_unvisited[x_conf_idx])
    return None # fail

    for n_iter in range(max_iter):
        pass