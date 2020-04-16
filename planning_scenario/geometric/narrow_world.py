import random
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
from matplotlib import patches

import config
import geometric_objects as gobj
from base_world import base_world

from solver import astar

class narrow_world(base_world):
    def __init__(self, num_robots, gap_width_range = None):
        if gap_width_range is None:
            gap_width_range = (config.ROBOT_RADIUS * 1.5, config.ROBOT_RADIUS * 2.5)
        assert gap_width_range[0] > config.ROBOT_RADIUS, "obstacle width too small!"
        self.num_robots = num_robots
        self.robots = []
        estimated_res = gap_width_range[0] + random.random() * (gap_width_range[1] - gap_width_range[0])
        self.initialize(estimated_res)
        while not (self.test(np.hstack([r.center for r in self.robots])) and self.test(np.hstack([r.goal for r in self.robots]))):
            self.initialize(estimated_res)
    
    def initialize(self, eps, random_init = True):
        self.robots = []
        # Initialize obstacles
        res_cnt = int(1. / eps)
        assert res_cnt > 6, "Robot size too large!"
        self.sample_pts = np.linspace(0, 1., res_cnt)
        self.true_eps = self.sample_pts[1] - self.sample_pts[0]
        self.horizontal_obs_lower_y = random.randint(1, res_cnt - 3)
        self.vertical_obs_lower_x = random.randint(1, res_cnt - 3)
        self.horizontal_first_gap_dis_lower_x = random.randint(0, self.vertical_obs_lower_x - 1)
        self.horizontal_second_gap_dis_lower_x = random.randint(self.vertical_obs_lower_x + 1, res_cnt - 2)
        self.vertical_obs_gap_dis_lower_y = random.randint(self.horizontal_obs_lower_y + 1, res_cnt - 2)
        horizontal_left_obstacle = gobj.Rectangle((0, self.sample_pts[self.horizontal_obs_lower_y]),
            (self.sample_pts[self.horizontal_first_gap_dis_lower_x], self.sample_pts[self.horizontal_obs_lower_y + 1]))
        horizontal_middle_obstacle = gobj.Rectangle((self.sample_pts[self.horizontal_first_gap_dis_lower_x + 1], self.sample_pts[self.horizontal_obs_lower_y]),
            (self.sample_pts[self.horizontal_second_gap_dis_lower_x], self.sample_pts[self.horizontal_obs_lower_y + 1]))
        horizontal_right_obstacle = gobj.Rectangle((self.sample_pts[self.horizontal_second_gap_dis_lower_x + 1], self.sample_pts[self.horizontal_obs_lower_y]),
            (self.sample_pts[-1], self.sample_pts[self.horizontal_obs_lower_y + 1]))
        vertical_lower_obstacle = gobj.Rectangle((self.sample_pts[self.vertical_obs_lower_x], self.sample_pts[0]),
            (self.sample_pts[self.vertical_obs_lower_x + 1], self.sample_pts[self.vertical_obs_gap_dis_lower_y]))
        vertical_higher_obstacle = gobj.Rectangle((self.sample_pts[self.vertical_obs_lower_x], self.sample_pts[self.vertical_obs_gap_dis_lower_y + 1]),
            (self.sample_pts[self.vertical_obs_lower_x + 1], self.sample_pts[-1]))
        self.obstacles = [horizontal_left_obstacle, horizontal_middle_obstacle, horizontal_right_obstacle,
            vertical_lower_obstacle, vertical_higher_obstacle]
        # Initialize robots
        if random_init:
            for i in range(self.num_robots):
                start_pt = np.random.random(size = (2,))
                goal_pt = np.random.random(size = (2,))
                self.robots.append(gobj.Robot(start_pt, config.ROBOT_RADIUS, goal_pt, i))
        else:
            pass

    def test(self, robot_conf):
        '''
        Test the given robot_conf is collision-free.

        Args:
            robot_conf: np.array of shape (self.num_robots * 2,) that gives a configuration of robots
        '''
        my_robot_conf = robot_conf.reshape((-1, 2))
        for o in self.obstacles:
            for i in range(my_robot_conf.shape[0]):
                cur_robot_center = my_robot_conf[i]
                robot_radius = self.robots[i].radius
                if o.robot_collides(cur_robot_center, robot_radius):
                    return False
        return True
            

    def plot(self, soln = None):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect = 'equal')
        for o in self.obstacles:
            o.draw_matplotlib(ax, alpha = 0.6)
        for r in self.robots:
            r.draw_matplotlib(ax)
        # for i in range(0,gridSize*gridSize): # plot occupancy grid
        #     if occGrid[i] == 0:
        #         plt.scatter(occGridSamples[i,0], occGridSamples[i,1], color="red", s=70, alpha=0.8)
        #     else:
        #         plt.scatter(occGridSamples[i,0], occGridSamples[i,1], color="green", s=70, alpha=0.8)
        plt.show()

    def solve(self, solver):
        pass

    def get_trainable_data(self):
        pass

if __name__ == "__main__":
    test_world = narrow_world(1)
    test_world.plot()