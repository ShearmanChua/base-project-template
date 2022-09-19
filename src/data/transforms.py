import os
import copy
import math
import random
import numpy as np
from typing import Callable, Optional, Tuple, List

import torch
from torch import Tensor




class UpperClamp():
    '''
    Clamps the values per column
    '''
    def __init__(self, upper_bound: List[float]) -> None:
        self.upper_bound = torch.Tensor(upper_bound).unsqueeze(0)
    
    def __call__(self, x: Tensor) -> Tensor:
        return torch.min(x, self.upper_bound)

class LowerClamp():
    '''
    Clamps the values per column
    '''
    def __init__(self, lower_bound: List[float]) -> None:
        self.lower_bound = torch.Tensor(lower_bound).unsqueeze(0)

    def __call__(self, x: Tensor) -> Tensor:
        return torch.max(x, self.lower_bound)

class Normalize():
    '''
    Normalizes each feature across time. 
    output will be between -1,1
    (sequence,features)
    '''
    def __init__(self, kwargs) -> None:
        '''
        norm_values has shape (1,num_features)
        '''

        upper_norm_values = kwargs['upper_norm_values']
        lower_norm_values = kwargs['lower_norm_values']

        upper_norm_values = np.expand_dims(np.array(upper_norm_values),axis=0)
        self.lower_norm_values = np.expand_dims(np.array(lower_norm_values),axis=0)
        self.range = upper_norm_values - self.lower_norm_values

    def __call__(self, x: Tensor) -> Tensor:
        '''
        input : (sequence,features)
        output : (sequence,features)
        ''' 
        x = x - self.lower_norm_values
        return (x/self.range)*2 - 1

class NormalizeZeroOne():
    '''
    Normalizes each feature across time. 
    output will be between 0,1
    (sequence,features)
    '''
    def __init__(self, norm_values: List[float]) -> None:
        '''
        norm_values has shape (1,num_features)
        '''
        upper_norm_values = np.expand_dims(np.array(upper_norm_values),axis=0)
        self.lower_norm_values = np.expand_dims(np.array(lower_norm_values),axis=0)
        self.range = upper_norm_values - self.lower_norm_values

    def __call__(self, x: Tensor) -> Tensor:
        '''
        input : (sequence,features)
        output : (sequence,features)
        '''
        x = x - self.lower_norm_values
        return (x/self.range)

class Rotate():
    '''
    Rotate the direction vectors across time
    '''
    def __init__(self, rotation_angle: List[float]) -> None:
        '''
        rotation_angle is a list of size 1

        user-determined on maximum angle allowed to rotate
        '''
        self.rotation_angle = rotation_angle[0]

    def rotate_pt(self, x: Tensor) -> Tensor:
        '''
        Direction vectors are rotated by rotation matrix
        note that longitude is not rotated
        note that first element here is change in lat, second is lon

        input : (1,features)
        output : (1,features)
        '''
        point_x, point_y = x[0], x[1]
        theta = np.radians(self.rotation_angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, s), (-s, c)))
        v = np.array((point_x, point_y))
        x[0], x[1] = np.dot(R, v)
        return x

    def __call__(self, x: Tensor) -> Tensor:
        '''
        input : (sequence,features)
        output : (sequence,features)
        '''
        rand_angle = random.randrange(0, 360, self.rotation_angle)
        return torch.stack([self.rotate_pt(x[i]) for i in range(x.shape[0])])

class RemoveLocation():
    '''
    Change inputs from lat, lon to delta lat and delta lon
    '''
    def __init__(self, args: List[float]) -> None:
        pass

    def __call__(self, x: Tensor) -> Tensor:
        '''
        To remove the first row of the tensor and subtract it with the original x,
        this gets the change /delta needed. eos is inputted as tensors of zeros
                
        input : (sequence,features)
        output : (sequence-1,features)
        '''
        tobe_subtracted = copy.deepcopy(x)[1:,:]
        tobe_subtracted = torch.cat((tobe_subtracted, torch.zeros(1, tobe_subtracted.shape[1])))
        x[:,2:] = 0
        change = tobe_subtracted - x
        return change[:-1,:]
class TimeEncoder():

    '''
    This is NOT the exact PE used in transformers, which uses index as position.
    this uses the time in seconds since the start of trajectory as position
    '''
    def __init__(self, n_dims):
        super().__init__()
        self.n_dims = n_dims
    
    def __call__(self, x):
        '''
        x is a time vector of size [max_len]
        '''
        # make embeddings relatively larger
        max_len = x.shape[0]
        pe = torch.zeros(max_len,self.n_dims)
        for i in range(0, self.n_dims, 2):
            pe[:,i] = torch.sin(x/ (10000 ** ((2 * i)/self.n_dims)))
            pe[:,i+1] = torch.cos(x/ (10000 ** ((2 * (i))/self.n_dims)))           
        return pe
        

class DateTimeEncoder():
    MAX_TIME = 24*60
    '''
    This generates the minute, hour, day,dayofweek,month features from timestamps
    '''
    def __init__(self):
        super().__init__()
        
    
    def __call__(self, x):
        '''
        x is a numpy datetime series of size [max_len]
        '''
        # make embeddings relatively larger
        time = self._cyclic_mapping(x.dt.hour.values*60 + x.dt.minute.values,self.MAX_TIME)
        
        
        
        day = self._cyclic_mapping(x.dt.day.values-1,31)
        dayofweek = self._cyclic_mapping(x.dt.dayofweek.values,7) #max 6
        month = self._cyclic_mapping(x.dt.month.values-1,12)

        stacked_x = np.concatenate([time,day,dayofweek,month],-1)
        
        return torch.from_numpy(stacked_x)

    def _cyclic_mapping(self,x,max_value):
        '''
        converts datetime into a cyclic format using sin and cos
        values must range from [0 to max_value)
        note that max_value will be mapped to the same as 0, so your max in data should be asypmtopically bounded by max_value
        values should start at 0
        '''
        x_norm = (x/max_value)*2*math.pi
        return np.stack([np.sin(x_norm),np.cos(x_norm)],-1)

class LimitLength():
    '''
    Limit the maximum length (if needed)
    Note this is included to reduce training time
    '''
    def __init__(self, max_length: List[float]) -> None:
        self.max_length = max_length[0]
    
    def __call__(self, x: Tensor) -> Tensor:
        '''
        input : (sequence,features)
        output : (reduced_sequence,features)
        '''
        return x[:min(x.shape[0], self.max_length), :]

TRANSFORM_MAPPER = {
    'LimitLength' : LimitLength,
    'RemoveLocation': RemoveLocation,
    'UpperClamp': UpperClamp,
    'LowerClamp': LowerClamp,
    'Normalize': Normalize,
    'NormalizeZeroOne': NormalizeZeroOne,
    'Rotate': Rotate
        }

def get_transforms(transforms_dict):
    x_transforms = []
    for k,params in transforms_dict.items():
        x_transforms.append(TRANSFORM_MAPPER[k](params))
    return x_transforms