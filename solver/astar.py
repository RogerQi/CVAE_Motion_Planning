import xxhash
import numpy as np
import numpy.linalg as npla
import heapq

h = xxhash.xxh64()

class node(object):
    def __init__(self, state, parent_node, g, h):
        self.parent = parent_node
        self.state = state
        self.g = g
        self.h = h
        self.f = self.g + self.h

    def __lt__(self, other):
        return self.f < other.f

    def __gt__(self, other):
        return self.f > other.f

    def __eq__(self, other):
        return self.f == other.f

    def __le__(self, other):
        return (self < other) or (self == other)
    
    def __ge__(self, other):
        return (self > other) or (self == other)

def get_np_arr_hash(np_arr):
    h.update(np_arr)
    ret = h.intdigest()
    h.reset()
    return ret

def astar_base(start_conf, adj_func, test_goal_func, test_cfree_func, g_func, h_func):
    '''
    Abstract A* solver.

    Args:
        start_conf: a numpy array representing start configuration. Assumed collision-free.
        adj_func: pointer to a function such that, when called with a configuration as parameter,
            it returns potential candidates to be explored. Note that the returned value is not
            necessarily collision free.
        test_goal_func: pointer to a function which returns True if given configuration
            can be considered as goal; False otherwise.
        test_cfree_func: pointer to a function which tests if given configuration is collision-free.
        g_func: cost function that gives the edge weight between two configuration
        h_func: heuristic function that computes the heuristic of any configuration. Note that
            it does not necessarily need to be admissible.
    '''
    assert start_conf.dtype == np.int
    initial_h = h_func(start_conf)
    frontier = [node(start_conf, None, 0, initial_h)]
    visited_dict = {}
    while len(frontier) > 0:
        cur_node = heapq.heappop(frontier)
        # test if goal
        if test_goal_func(cur_node.state):
            # Back trace to extract node
            ret = []
            while cur_node is not None:
                ret.append(cur_node.state)
                cur_node = cur_node.parent
            # reverse to start
            ret = ret[::-1]
            return ret
        # Keep expanding...
        for new_state in adj_func(cur_node.state):
            state_hash_key = get_np_arr_hash(new_state)
            if state_hash_key in visited_dict:
                continue # Already visited this node
            if not test_cfree_func(new_state):
                continue # collision detected!
            edge_cost = g_func(cur_node.state, new_state)
            goal_h = h_func(new_state)
            new_node = node(new_state, cur_node, cur_node.g + edge_cost, goal_h)
            visited_dict[get_np_arr_hash(new_node.state)] = True
            heapq.heappush(frontier, new_node)
    return None
    

if __name__ == "__main__":
    a = node(np.ones((4,)), None, 1, 1)
    b = node(np.ones((4,)), None, 0, 0.5)
    test_dict = {}
    test_dict[get_np_arr_hash(a.state)] = True
    print(a.state.tostring().hex())
    print(b.state.tostring().hex())
    print(a.__hash__())
    print(b.__hash__())
    try:
        test_dict[get_np_arr_hash(b.state)]
        print("success")
    except KeyError:
        print("fail")