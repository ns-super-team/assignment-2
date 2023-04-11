import numpy as np

# https://stackoverflow.com/a/73704579
class EarlyStopper:
    def __init__(self, patience=1, delta=0, mode='min'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_value = np.inf
        self.max_value = 0
        self.mode = mode

    def __call__(self, value): 
        if self.mode == 'min':
            if value < self.min_value:
                self.min_value = value
                self.counter = 0     
            elif value > (self.min_value + self.delta):
                self.counter += 1
                print(f'Early stopping: {self.counter}/{self.patience}')
                
        elif self.mode == 'max':
            if value > self.max_value:
                self.max_value = value
                self.counter = 0     
            elif value < (self.max_value - self.delta):
                self.counter += 1
                print(f'Early stopping: {self.counter}/{self.patience}')
                
        if self.counter >= self.patience:
            return True
        
        # print((self.min_value + self.min_delta))
        
        return False