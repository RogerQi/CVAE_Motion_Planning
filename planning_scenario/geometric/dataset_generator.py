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
            self.constructor_params = (self.num_robot,)
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
    n_datapoints = 1000
    num_robots = 1
    max_obstacle_cnt = 10
    my_gen = geometric_data_gen(num_robots, max_obstacle_cnt, "narrow")
    gen_time_sum = 0
    solve_time_sum = 0
    gen_start = time.time()
    total_X = []
    total_C = []
    with tqdm(total = n_datapoints) as pbar:
        for w in my_gen.infinite_gen(n_datapoints):
            pbar.update(1)
            gen_time_sum += (time.time() - gen_start)
            solve_start = time.time()
            soln = w.solve("astar")
            if soln is None: continue
            solve_time_sum += (time.time() - solve_start)
            all_data = w.get_trainable_data(best_soln = soln, sample_interval = 50)
            for x, c in all_data:
                total_C.append(c)
                total_X.append(x)
            gen_start = time.time()
    total_X = np.vstack(total_X)
    total_C = np.vstack(total_C)
    print("X shape: {}".format(total_X.shape))
    print("C shape: {}".format(total_C.shape))
    print("Generate {0} maps with {1} robots and max {2} obstacles.".format(n_trial, num_robots, max_obstacle_cnt))
    print("Average time to generate a map is {0}.".format(gen_time_sum / n_trial))
    print("Average time to solve a map with astar is {0}.".format(solve_time_sum / n_trial))
    np.save("dataset_x.npy", total_X, allow_pickle = False, fix_imports = False)
    np.save("dataset_c.npy", total_C, allow_pickle = False, fix_imports = False)
    print("Generated data saved to dataset_x.npy and dataset_C.npy.")
    assert total_X.shape[0] == total_C.shape[0]

if __name__ == "__main__":
    main()