import numpy as np

# https://stackoverflow.com/a/73704579
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_value = np.inf

    def __call__(self, value): 
        if value < self.min_value:
            self.min_value = value
            self.counter = 0     
        elif value > (self.min_value + self.min_delta):
            self.counter += 1
            print(f'Early stopping: {self.counter}/{self.patience}')
            
            if self.counter >= self.patience:
                return True
        
        # print((self.min_value + self.min_delta))
        
        return False