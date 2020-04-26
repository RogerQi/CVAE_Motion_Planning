import sys
import os
import time
import random
import numpy as np
from multiprocessing import Process, Lock, Manager

from tqdm import tqdm

import random_world
import narrow_world

class general_worker(object):
    worker_id = 0

    def __init__(self, func, params, job_num):
        self.worker_id = general_worker.worker_id
        general_worker.worker_id += 1
        self.func = func
        self.params = params
        self.job_num = job_num
    
    def start(self, lock, buffer):
        print("Worker {} started.".format(self.worker_id))
        i = 0
        while i < self.job_num:
            cur_world = self.func(*self.params)
            soln = cur_world.solve("astar")
            if soln is None: continue
            all_data = cur_world.get_trainable_data(soln = soln, sample_interval = 20)
            lock.acquire()
            buffer.append(all_data)
            lock.release()
            i += 1

class geometric_data_gen(object):
    def __init__(self, num_worker, num_robot, max_num_obstacles, world_type):
        assert world_type in ["narrow", "random_rect", "random_circle"]
        assert num_worker >= 1 and isinstance(num_worker, int)
        self.num_robot = num_robot
        self.num_worker = num_worker
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
        manager = Manager()
        my_workers = []
        gen_buffer = manager.list()
        worker_lock = Lock()
        work_assigned = 0
        for i in range(self.num_worker - 1):
            job_num = desired_num // self.num_worker
            work_assigned += job_num
            my_workers.append(general_worker(self.callable_constructor, self.constructor_params,
                job_num))
        # last worker
        my_workers.append(general_worker(self.callable_constructor, self.constructor_params,
                desired_num - work_assigned))
        processes = []
        for i in range(self.num_worker):
            p = Process(target = my_workers[i].start, args = (worker_lock, gen_buffer))
            processes.append(p)
            p.start()
        returned_sample_cnt = 0
        while returned_sample_cnt < desired_num:
            while not gen_buffer:
                time.sleep(0.1)
            worker_lock.acquire()
            yield gen_buffer.pop(0)
            worker_lock.release()
            returned_sample_cnt += 1
        # for i in range(desired_num):
        #     yield self.callable_constructor(*self.constructor_params)
        for p in processes:
            p.join()

def main():
    n_datapoints = 10000
    num_robots = 1
    num_worker = 8
    max_obstacle_cnt = 10
    my_gen = geometric_data_gen(num_worker, num_robots, max_obstacle_cnt, "narrow")
    gen_time_sum = 0
    solve_time_sum = 0
    total_X = []
    total_C = []
    with tqdm(total = n_datapoints) as pbar:
        for all_data in my_gen.infinite_gen(n_datapoints):
            pbar.update(1)
            for x, c in all_data:
                total_C.append(c)
                total_X.append(x)
    total_X = np.vstack(total_X)
    total_C = np.vstack(total_C)
    print("X shape: {}".format(total_X.shape))
    print("C shape: {}".format(total_C.shape))
    print("Generate {0} maps with {1} robots and max {2} obstacles.".format(n_datapoints, num_robots, max_obstacle_cnt))
    np.save("dataset_x.npy", total_X, allow_pickle = False, fix_imports = False)
    np.save("dataset_c.npy", total_C, allow_pickle = False, fix_imports = False)
    print("Generated data saved to dataset_x.npy and dataset_C.npy.")
    assert total_X.shape[0] == total_C.shape[0]

if __name__ == "__main__":
    main()