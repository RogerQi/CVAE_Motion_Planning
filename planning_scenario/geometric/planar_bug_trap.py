import sys
import os
import random
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt

from geometric_objects import Rectangle, Circle, Robot, tilted_rect_robot

def add_path(custom_path):
    if custom_path not in sys.path: sys.path.insert(0, custom_path)

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..', '..', 'solver')
add_path(lib_path)

from base_world import base_world

rod_robot_length = 0.1
rod_robot_width = 0.05
left_bound = 0.2
right_bound = 0.8
upper_bound = 0.8
lower_bound = 0.2
passage_width = 0.2
internal_free_length = 0.25
rect_thickness = 0.05

class planar_bug_trap_world(base_world):
    '''
    __________________
    |                 |
    |                 |
    |      ___________|
    |
    |
    |      ___________           
    |                 |
    |                 |
    |_________________|
    '''
    def __init__(self, random_init = False):
        self.initialize(random_init)
    
    def initialize(self, random_init):
        # Compute some numbers
        right_rect_length = (upper_bound - lower_bound - passage_width) / 2
        # Construct rectangles
        left_rect = Rectangle((left_bound, lower_bound), (left_bound + rect_thickness, upper_bound))
        upper_rect = Rectangle((left_bound, upper_bound - rect_thickness), (right_bound, upper_bound))
        lower_rect = Rectangle((left_bound, lower_bound), (right_bound, lower_bound + rect_thickness))
        right_lower_rect = Rectangle((right_bound - rect_thickness, lower_bound), (right_bound, lower_bound + right_rect_length))
        right_upper_rect = Rectangle((right_bound - rect_thickness, upper_bound - right_rect_length - rect_thickness), (right_bound, upper_bound))
        middle_lower_rect = Rectangle((left_bound + internal_free_length + rect_thickness, lower_bound + right_rect_length),
                                    (right_bound, lower_bound + right_rect_length + rect_thickness))
        middle_upper_rect = Rectangle((left_bound + internal_free_length + rect_thickness, upper_bound - right_rect_length - rect_thickness),
                                    (right_bound, upper_bound - right_rect_length))
        self.obstacles = [left_rect, upper_rect, lower_rect, right_lower_rect, right_upper_rect, middle_lower_rect, middle_upper_rect]
        self.robot = None

    def test(self, robot_conf):
        pass

    def plot(self, _ax = None, soln = None):
        ax = _ax
        if _ax is None:
            fig = plt.figure(figsize = (8, 8))
            ax = fig.add_subplot(111, aspect = 'equal')
        for o in self.obstacles:
            o.draw_matplotlib(ax, alpha = 0.6)
        # for r in self.robots:
        #     r.draw_matplotlib(ax)
        if _ax is None:
            plt.show()

    def get_trainable_data(self, soln = None, sample_interval = 1):
        # Get Conditional
        # Get solution
        # Get conditional
        # IMPORTANT: THIS MUST BE HANDLED CAREFULLY!
        # current scheme: concatenate initial/goal/encoded gap
        # gap encoding: (xmin, ymin, xmax, ymax)
        pass

if __name__ == '__main__':
    test_world = planar_bug_trap_world(False)
    test_world.plot()