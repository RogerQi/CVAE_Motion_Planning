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

rod_robot_length = 0.15
rod_robot_width = 0.03
left_bound = 0.2
right_bound = 0.8
upper_bound = 0.8
lower_bound = 0.2
passage_width = 0.2
internal_free_length = 0.25
rect_thickness = 0.05

def get_rod_robot_conf(num_robot):
    return np.random.uniform(low = (0, 0, 0), high = (1, 1, np.pi), size = (num_robot, 3))

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
    def __init__(self, num_robot, random_init = False):
        assert num_robot == 1, "Only one robot is supported now"
        self.num_robot = num_robot
        self.initialize(random_init)
        while not self.test(self.start_conf):
            self.initialize(random_init)
    
    def initialize(self, random_init):
        self.obstacles = []
        self.robots = []
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
        self.start_conf = []
        for r in range(self.num_robot):
            cur_conf = get_rod_robot_conf(1)
            unscaled_theta = cur_conf[0, 2]
            scaled_theta = unscaled_theta / np.pi
            cur_center = (cur_conf[0,0], cur_conf[0,1])
            cur_robot = tilted_rect_robot(cur_center, rod_robot_width, rod_robot_length, unscaled_theta)
            self.robots.append(cur_robot)
            self.start_conf += [cur_conf[0, 0], cur_conf[0, 1], scaled_theta]
        self.start_conf = np.array(self.start_conf)

    def test(self, robot_conf):
        '''
        Test the given robot_conf is collision-free.

        Args:
            robot_conf: np.array of shape (self.num_robots * 3,) that gives a configuration of robots
        '''
        my_robot_conf = robot_conf.copy().reshape((-1, 3))
        my_robot_conf[:,2] *= np.pi # Scale
        for robot_loc in my_robot_conf:
            if robot_loc[0] < 0 or robot_loc[0] > 1:
                return False # Out of bound
            if robot_loc[1] < 0 or robot_loc[1] > 1:
                return False
        for i in range(my_robot_conf.shape[0]):
            cur_robot_center = my_robot_conf[i,0:2]
            cur_theta = my_robot_conf[i,2]
            cur_robot_pt_set = tilted_rect_robot.get_pt_set(cur_robot_center, rod_robot_width, rod_robot_length, cur_theta)
            for o in self.obstacles:
                assert isinstance(o, Rectangle)
                if o.tilted_rect_robot_collides(cur_robot_pt_set):
                    return False
        for i in range(my_robot_conf.shape[0]):
            for r in range(len(self.robots)):
                if i == r: continue # current robot
                if self.robots[r].robot_robot_collides(my_robot_conf[i], my_robot_conf[r]):
                    return False # collide
        return True

    def plot(self, _ax = None, soln = None):
        ax = _ax
        if _ax is None:
            fig = plt.figure(figsize = (8, 8))
            ax = fig.add_subplot(111, aspect = 'equal')
        for o in self.obstacles:
            o.draw_matplotlib(ax, alpha = 0.6)
        for r in self.robots:
            r.draw_matplotlib(ax)
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
    # np.random.seed(128)
    test_world = planar_bug_trap_world(1, False)
    print("Start conf: {}".format(test_world.start_conf))
    test_world.plot()