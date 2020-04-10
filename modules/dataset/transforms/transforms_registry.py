class registry(dict):
    '''
    A helper class for registering transform methods to Python dictionary.
    '''
    def __init__(self, *args, **kwargs):
        super(registry, self).__init__(*args, **kwargs)
    
    def register(self, func):
        '''
        Should only be used as decorator!
        '''
        # Function factory
        func_name = func.__name__
        self[func_name] = func
        return func