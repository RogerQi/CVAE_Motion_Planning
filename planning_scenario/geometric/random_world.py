import random
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt

from config import ROBOT_RADIUS
from geometric_objects import Rectangle, Circle, Robot
from base_geometric_world import base_geometric_world

class random_world(base_geometric_world):
    '''
    World that organizes everything for geometric planning problem.
    '''
    def __init__(self, num_robots, max_num_obstacles, obstacle_type, h = 1.0, w = 1.0):
        assert h == 1.0 and w == 1.0 # should be scale to 1s
        assert obstacle_type in ["circle", "rectangle"]
        self.h, self.w = h, w
        self.obstacles = []
        self.robots = []
        self.soln_dict = {}
        self.num_robots = num_robots
        self.initialize(num_robots, max_num_obstacles, obstacle_type)
        while not (self.test(np.hstack([r.center for r in self.robots])) and self.test(np.hstack([r.goal for r in self.robots]))):
            self.initialize(num_robots, max_num_obstacles, obstacle_type)

    def initialize(self, num_robots, max_num_obstacles, obstacle_type):
        # Initialize obstacles
        self.obstacles = []
        self.robots = []
        num_obs = random.randint(1, max_num_obstacles)
        for j in range(num_obs):
            if obstacle_type == "rectangle":
                bmin = [1221, 0]
                bmax = [0, 0]
                while bmin[0] >= bmax[0] or bmin[1] >= bmax[1]:
                    min_x = round(random.random(), 3)
                    min_y = round(random.random(), 3)
                    width = random.random() * (1 - min_x - 1e-5)
                    height = random.random() * (1 - min_y - 1e-5)
                    bmin = (min_x, min_y)
                    bmax = (min_x + width, min_y + height)
                self.obstacles.append(Rectangle(bmin, bmax))
            else:
                rand_center = (round(random.random(), 3), round(random.random(), 3))
                radius_limit = [rand_center[0], 1 - rand_center[0], rand_center[1], 1 - rand_center[1]]
                rand_radius = round(random.uniform(0, min(radius_limit)), 3)
                self.obstacles.append(Circle(rand_center,rand_radius))
        # Initialize start state and goal state
        for i in range(num_robots):
            start_pt = np.random.random(size = (2,))
            goal_pt = np.random.random(size = (2,))
            self.robots.append(Robot(start_pt, ROBOT_RADIUS, goal_pt, i))

    def test_one(self, robot_id, robot_loc, test_robot_collision):
        '''
        Test if the given robot_loc is free of collision. Return True if no collision;
        Otherwise return False

        Args:
            robot_id: an integer denoting which robot it's referring to
            robot_loc: a 2D vector denoting the location of the robot
            test_robot_collision: a boolean indicating whether collision with other robot needs to be tested.
        '''
        assert robot_loc.shape == (2,)
        if robot_loc[0] < 0 or robot_loc[0] > self.w:
            return False # Out of bound
        if robot_loc[1] < 0 or robot_loc[1] > self.h:
            return False
        cur_test_robot = self.robots[robot_id]
        # Test obstacle collision
        for o in self.obstacles:
            if o.robot_collides(robot_loc, cur_test_robot.radius):
                return False
        if test_robot_collision:
            for r in range(len(self.robots)):
                if r == robot_id: continue # cur testbot
                if self.robots[r].robot_collides(robot_loc, cur_test_robot.radius):
                    return False
        return True
    
    def test(self, robot_conf):
        '''
        Test the given robot_conf is collision-free.

        Args:
            robot_conf: np.array of shape (self.num_robots * 2,) that gives a configuration of robots
        '''
        my_robot_conf = robot_conf.reshape((-1, 2))
        for i in range(my_robot_conf.shape[0]):
            if not self.test_one(i, my_robot_conf[i], True):
                return False
        return True
    
    def get_trainable_data(self, soln = None, sample_interval = 1):
        '''
        Get data that can be used to train the motion planning CVAE

        return
            ret: a list that contains multiple data points.
                Each data point can be interpreted in (X, cond) as in CVAE.
        '''
        assert self.soln_dict, "No solution? call world.solve first."
        # Get conditional
        # IMPORTANT: THIS MUST BE HANDLED CAREFULLY!
        # current scheme: concatenate initial/goal/encoded obstacles
        # obstacle encoding:
        #   - rectangle: upperleft, lowerright
        #   - circle: center + radius
        # RECTANGLE AND CIRCLE MUST NOT APPEAR TOGETHER AS IT CONFUSES CVAE!!!
        initial_conf = []
        goal_conf = []
        for r in self.robots:
            cur_pos = r.center
            goal_pos = r.goal
            initial_conf.append(cur_pos)
            goal_conf.append(goal_pos)
        initial_conf = np.array(initial_conf).flatten()
        goal_conf = np.array(goal_conf).flatten()
        obstacle_encode = []
        for obs in self.obstacles:
            params = obs.get_parameter()
            obstacle_encode.append(params)
        obstacle_encode = np.array(obstacle_encode).flatten()
        cond = [initial_conf, goal_conf, obstacle_encode]
        cond = np.concatenate(cond).flatten()
        # Get solution
        if soln is None:
            soln = self.get_best_soln()
        soln = np.array(soln).reshape((-1, self.num_robots * 2))
        ret = []
        for i in range(0, len(soln), sample_interval):
            ret.append((soln[i], cond))
        return ret

if __name__ == '__main__':
    test_world = random_world(1, 10, "rectangle")
    test_world.plot()
    soln = test_world.solve("FMT*")
    test_world.plot(soln = soln)
    data = test_world.get_trainable_data()
    print(data[0])