import sys
import os
import time
import random
import numpy as np

from tqdm import tqdm

import random_world
import narrow_world

class geometric_data_gen(object):
    def __init__(self, num_robot, max_num_obstacles, world_type):
        assert world_type in ["narrow", "random_rect", "random_circle"]
        self.num_robot = num_robot
        self.max_num_obstacles = max_num_obstacles
        if world_type == "narrow":
            self.callable_constructor = narrow_world.narrow_world
            self.constructor_params = (self.num_robot, self.max_num_obstacles)
        elif world_type == "random_rect":
            self.callable_constructor = random_world.random_world
            self.constructor_params = (self.num_robot, self.max_num_obstacles, "rectangle")
        elif world_type == "random_circle":
            self.callable_constructor = random_world.random_world
            self.constructor_params = (self.num_robot, self.max_num_obstacles, "circle")
        else:
            raise NotImplementedError
    
    def infinite_gen(self, desired_num = np.iinfo(np.int).max):
        for i in range(desired_num):
            yield self.callable_constructor(*self.constructor_params)

def main():
    n_trial = 10
    num_robots = 2
    max_obstacle_cnt = 10
    my_gen = geometric_data_gen(num_robots, max_obstacle_cnt, "random_rect")
    gen_time_sum = 0
    solve_time_sum = 0
    gen_start = time.time()
    with tqdm(total = n_trial) as pbar:
        for w in my_gen.infinite_gen(n_trial):
            pbar.update(1)
            gen_time_sum += (time.time() - gen_start)
            solve_start = time.time()
            w.solve("astar")
            solve_time_sum += (time.time() - solve_start)
            gen_start = time.time()
    print("Generate {0} maps with {1} robots and max {2} obstacles.".format(n_trial, num_robots, max_obstacle_cnt))
    print("Average time to generate a map is {0}.".format(gen_time_sum / n_trial))
    print("Average time to solve a map with astar is {0}.".format(solve_time_sum / n_trial))

if __name__ == "__main__":
    main()