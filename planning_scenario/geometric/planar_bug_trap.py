import sys
import os
import random
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt

from geometric_objects import Rectangle, Circle, Robot
from base_world import base_world

def add_path(custom_path):
    if custom_path not in sys.path: sys.path.insert(0, custom_path)

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..', '..', 'solver')
add_path(lib_path)

class planar_bug_trap_world(base_world):
    def __init__(self, num_robots, gap_width_range = None, random_init = False):
        pass
    
    def initialize(self, eps, random_init):
        pass

    def test(self, robot_conf):
        pass

    def get_trainable_data(self, soln = None, sample_interval = 1):
        # Get Conditional
        # Get solution
        # Get conditional
        # IMPORTANT: THIS MUST BE HANDLED CAREFULLY!
        # current scheme: concatenate initial/goal/encoded gap
        # gap encoding: (xmin, ymin, xmax, ymax)
        pass
