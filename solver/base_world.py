import sys
import os

class base_world(object):
    # Abstact world used by solvers
    def __init__(self):
        '''
        Initialize the world.
        '''
        raise NotImplementedError

    def test(self, conf):
        '''
        Return if a given configuration is collision-free.
            Return True if collision-free; False otherwise.

        Args:
            - conf: an indexable object representing configuration
        '''
        raise NotImplementedError

    def plot(self):
        '''
        Visualize the world.
        '''
        raise NotImplementedError