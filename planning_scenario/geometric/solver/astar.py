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

    def __hash__(self):
        h.update(self.state)
        ret = h.intdigest()
        h.reset()
        return ret

def get_discrete_coordinate(continuous_coord, eps, h = 1., w = 1.):
    discrete_coord = continuous_coord.copy().reshape((-1, 2))
    discrete_coord[:,0] *= int(w / eps)
    discrete_coord[:,1] *= int(h / eps)
    return discrete_coord.astype(np.int).reshape((-1,))

def get_continous_coordinate(discrete_coord, eps, h = 1., w = 1.):
    continuous_coord = discrete_coord.astype(np.float, copy = True).reshape((-1, 2))
    continuous_coord[:,0] *= (eps / w)
    continuous_coord[:,1] *= (eps / h)
    return continuous_coord.reshape((-1,))

def euclidean_dist(cur_state, other_state):
    cur_state = cur_state.reshape((-1, 2))
    other_state = other_state.reshape((-1, 2))
    diff_state = cur_state - other_state
    return npla.norm(diff_state, axis = 1)

def heuristic(state, goal_state):
    # Change this function as long as it stays admissible
    # Heuristic is geodesic in an unconstraint ND space
    return np.sum(euclidean_dist(state, goal_state))

def construct_direction(total_entries):
    '''
    Construct a list of offset vector for expansion.
    Possible element: -1, 0, 1
    '''
    assert isinstance(total_entries, int)
    if total_entries == 1:
        return [[-1], [0], [1]]
    else:
        offset_vecs = construct_direction(total_entries - 1)
        ret = []
        for l in offset_vecs:
            ret += [l + [-1], l + [0], l + [1]]
        return ret

def get_np_arr_hash(np_arr):
    h.update(np_arr)
    ret = h.intdigest()
    h.reset()
    return ret

def astar_solve(the_world, eps = 0.01):
    '''
    On-the-fly astart solver

    Args:
        the_world: a world object as defined in world.py
        eps: discretization resolution
    
    Return:
        path: a sequence of collision-free states from current states of robot
            to the goal state. The edit distance between each two adjacent state
            in the sequence is at most 4 ($\pm 1$ for each term).
    '''
    #####################
    # Setup
    #####################
    goal_tolerance = 3
    num_robots = len(the_world.robots)
    robot_radius = the_world.robots[0].radius
    # State indexing: [robot_id, x/y]
    cur_state = np.hstack([r.center for r in the_world.robots])
    goal_state = np.hstack([r.goal for r in the_world.robots])
    cur_state = get_discrete_coordinate(cur_state, eps)
    goal_state = get_discrete_coordinate(goal_state, eps)
    initial_h = heuristic(cur_state, goal_state)
    frontier = [node(cur_state, None, 0, initial_h)] # to be used in heapq
    visited_dict = {}
    possible_direction = construct_direction(cur_state.shape[0])
    possible_direction = [np.array(arr, dtype = "int") for arr in possible_direction]
    min_h = 999999999
    while len(frontier) > 0:
        cur_node = heapq.heappop(frontier)
        if cur_node.h < min_h:
            min_h = cur_node.h
        # test if goal
        if npla.norm(cur_node.state - goal_state) <= goal_tolerance:
            # Back trace to extract node
            ret = []
            while cur_node is not None:
                ret.append(cur_node.state)
                cur_node = cur_node.parent
            ret = ret[::-1]
            return [get_continous_coordinate(s, eps) for s in ret]
        # Keep expanding...
        for offset_dir in possible_direction:
            new_state = cur_node.state + offset_dir
            new_state_in_continuous_coord = get_continous_coordinate(new_state, eps)
            if not the_world.test(new_state_in_continuous_coord):
                continue # collision detected!
            dist = euclidean_dist(cur_node.state, new_state)
            dist = np.max(dist)
            goal_dist = heuristic(new_state, goal_state)
            new_node = node(new_state, cur_node, cur_node.g + dist, goal_dist)
            try:
                visited_dict[get_np_arr_hash(new_node.state)]
                continue
            except KeyError:
                pass
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