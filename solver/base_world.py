import sys
import os

class base_world(object):
    # Abstact world used by solvers
    def __init__(self):
        raise NotImplementedError

    def test(self, conf):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError