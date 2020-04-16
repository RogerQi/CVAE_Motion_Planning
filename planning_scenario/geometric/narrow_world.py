import random
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt

import config
from base_world import base_world

from solver import astar

class narrow_world(base_world):
    def __init__(self, gap_width_range = None):
        if gap_width_range is None:
            gap_width_range = (config.ROBOT_RADIUS * 1.5, config.ROBOT_RADIUS * 2.5)
    
    def test(self, robot_conf):
        pass

    def plot(self, soln = None):
        pass

    def solve(self, solver):
        pass

    def get_trainable_data(self):
        pass