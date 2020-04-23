import numpy as np
import numpy.linalg as npla                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

def rrt_base(start_conf, sampling_func, test_goal_func, test_cfree_func):
    '''
    Abstract naive RRT solver.

    Args:
        start_conf: a numpy array representing start configuration. Assumed collision-free.
        sampling_func: a function which samples a point in the configuration space.
            Sidenote: this can be non-uniform!
        test_goal_func: pointer to a function which returns True if given configuration
            can be considered as goal; False otherwise.
        test_cfree_func: pointer to a function which tests if given configuration is collision-free.
    '''
    assert start_conf.dtype == np.float
