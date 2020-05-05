import random
import numpy as np
import numpy.linalg as npla

import sys
import os

def add_path(custom_path):
    if custom_path not in sys.path: sys.path.insert(0, custom_path)

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..', '..', 'solver')
add_path(lib_path)

from astar import astar_base
from rrt import rrt_base
from bidirectional_rrt import bidirectional_rrt_base
from rrt_star import bidirectional_rrt_star_base
from fmt_star import fmt_star_base

def euclidean_dist(cur_state, other_state):
    cur_state = cur_state.reshape((-1, 3))
    other_state = other_state.reshape((-1, 3))
    diff_state = cur_state - other_state
    return npla.norm(diff_state, axis = 1)

def bidirectional_rrt_star_solve(the_world, sample_segment = 100):
    num_robots = len(the_world.robots)
    initial_state = the_world.start_conf
    goal_state = the_world.goal_conf
    state_shape = initial_state.shape

    def sampling_func():
        return np.random.random(size = state_shape).reshape(state_shape)
    
    def interpolate_func(state_a, state_b):
        return np.linspace(state_a, state_b, sample_segment)
    
    def metric_func(state_a, state_b):
        dist = euclidean_dist(state_a, state_b)
        return np.max(dist)
        
    def test_cfree_func(state_a):
        return the_world.test(state_a)
    
    soln = bidirectional_rrt_star_base(initial_state, goal_state, sampling_func, interpolate_func,
        metric_func, test_cfree_func, k = 100)
    if soln is not None:
        return soln
    else:
        return None

def bidirectional_rrt_solve(the_world, sample_segment = 100):
    num_robots = len(the_world.robots)
    initial_state = the_world.start_conf
    goal_state = the_world.goal_conf
    state_shape = initial_state.shape

    def sampling_func():
        return np.random.random(size = state_shape)
    
    def interpolate_func(state_a, state_b, reverse = False):
        if reverse:
            state_a[2], state_b[2] = state_b[2], state_a[2]
        return np.linspace(state_a, state_b, sample_segment)
    
    def metric_func(state_a, state_b):
        dist = euclidean_dist(state_a, state_b)
        return np.max(dist)
        
    def test_cfree_func(state_a):
        return the_world.test(state_a)
    
    soln = bidirectional_rrt_base(initial_state, goal_state, sampling_func, interpolate_func,
        metric_func, test_cfree_func, k = 100)
    if soln is not None:
        return soln
    else:
        return None