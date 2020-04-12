import random
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt

import config
import geometric_objects as gobj

from solver import astar

class world(object):
    '''
    World that organizes everything for geometric planning problem.
    '''
    def __init__(self, num_robots, max_num_obstacles, h = 1.0, w = 1.0):
        assert h == 1.0 and w == 1.0 # should be scale to 1s
        self.h, self.w = h, w
        self.obstacles = []
        self.robots = []
        self.num_robots = num_robots
        self.initialize(num_robots, max_num_obstacles)
        while not (self.test(np.hstack([r.center for r in self.robots])) and self.test(np.hstack([r.goal for r in self.robots]))):
            self.initialize(num_robots, max_num_obstacles)

    def initialize(self, num_robots, max_num_obstacles):
        # Initialize obstacles
        self.obstacles = []
        self.robots = []
        num_obs = random.randint(1, max_num_obstacles)
        for j in range(num_obs):
            if random.randint(0,1):
                bmin = [1221, 0]
                bmax = [0, 0]
                while bmin[0] >= bmax[0] or bmin[1] >= bmax[1]:
                    min_x = round(random.random(), 3)
                    min_y = round(random.random(), 3)
                    width = random.random() * (1 - min_x - 1e-5)
                    height = random.random() * (1 - min_y - 1e-5)
                    bmin = (min_x, min_y)
                    bmax = (min_x + width, min_y + height)
                self.obstacles.append(gobj.Rectangle(bmin, bmax))
            else:
                rand_center = (round(random.random(), 3), round(random.random(), 3))
                radius_limit = [rand_center[0], 1 - rand_center[0], rand_center[1], 1 - rand_center[1]]
                rand_radius = round(random.uniform(0, min(radius_limit)), 3)
                self.obstacles.append(gobj.Circle(rand_center,rand_radius))
        # Initialize start state and goal state
        for i in range(num_robots):
            start_pt = np.random.random(size = (2,))
            goal_pt = np.random.random(size = (2,))
            self.robots.append(gobj.Robot(start_pt, config.ROBOT_RADIUS, goal_pt, i))

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

    def plot(self, soln = None):
        '''
        Use matplotlib to plot current world map for debugging purpose.

        Args
            soln: a list of numpy sequence denoting the path of robots
        '''
        plt.figure(figsize=(5, 5))
        plt.axis('equal')
        plt.xlim(0,1)
        plt.ylim(0,1)
        for o in self.obstacles:
            o.draw_matplotlib(plt.gca())
        for r in self.robots:
            r.draw_matplotlib(plt.gca())
        if soln is not None:
            soln = np.array(soln).reshape((-1, self.num_robots, 2))
            for i in range(self.num_robots):
                plt.plot(soln[:,i,0], soln[:,i,1])
        plt.show()

    def solve(self, solver):
        '''
        Return solution using specified solver
        '''
        assert solver in ["rrt", "prm", "astar", "fmt"]
        if solver == "astar":
            ret = astar.astar_solve(self)
            if ret is None:
                print("No solution found!")
            return ret

if __name__ == '__main__':
    test_world = world(2, 10)
    test_world.plot()
    astar_soln = test_world.solve("astar")
    test_world.plot(astar_soln)
    