from typing import List, Callable
import os
import json
import copy


import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import dask.dataframe as dd

from .transforms import TimeEncoder, DateTimeEncoder
from . import transforms
from typing import Dict


class FlightDataset(Dataset):
    """
    Flight_Dataset Object
    """

    # note that _ is padding to reach charlimit and -  is unknown
    CALLSIGN_TOKENS = '0123456789abcdefghijklmnopqrstuvwxyz-_'
    CALLSIGN_CHAR2IDX = {v: i for i, v in enumerate(CALLSIGN_TOKENS)}

    # note that _ is padding to reach charlimit and -  is unknown
    MODE3_TOKENS = '0123456789-_'
    MODE3_CHAR2IDX = {v: i for i, v in enumerate(MODE3_TOKENS)}

    def _map_callsign(self, string):
        return [self.CALLSIGN_CHAR2IDX[i] for i in string]

    def _map_mode3(self, string):
        return [self.MODE3_CHAR2IDX[i] for i in string]

    def __init__(self, datapath: str, features: List[str], label: str, mode3_column: str, callsign_column: str, mode: str, transforms_dict: Dict[str, List[int]], time_encoding_dims: int) -> None:
        """
        Initialises the Flight_Dataset object.
        :param dataset: numpy array of the dataset (in function_class.py)
        :param data_labels: labels of the dataset given
        :param labels_dct: dictionary containing encoding of classes
        :param mode: train or valid or test; assign encodings of labels
        """

        self.root_folder = os.path.join(datapath, mode)
        with open(os.path.join(self.root_folder, 'metadata.json')) as infile:
            self.metadata = json.load(infile)
        self.features = features
        self.label = label
        self.labels_map = {l: i for i, l in enumerate(self.metadata['labels'])}
        self.n_classes = len(self.metadata['labels'])
        self.labels_map['start'] = len(self.metadata['labels'])
        self.labels_map['pad'] = len(self.metadata['labels']) + 1

        self.callsign_column = callsign_column
        self.mode3_column = mode3_column
        # self.labels_map['unknown'] = 2
        self.mode = mode
        self.data = dd.read_parquet(os.path.join(self.root_folder, 'data.parquet'),
                                    columns=self.features +
                                    [self.label] + [self.callsign_column,
                                                    self.mode3_column, 'datetime'],
                                    engine='fastparquet')  # this is lazy loading, its not actually loading into memory
        transforms_list = transforms.get_transforms(transforms_dict)
        self.transforms = Compose(transforms_list)
        self.length_mapper = {}
        self.idx_to_track_id = {}
        idx = 0
        for k, v in self.metadata['length'].items():
            int_key = int(k)
            self.length_mapper[int_key] = v
            self.idx_to_track_id[idx] = int_key
            idx = idx + 1

        self.time_encoding_dims = time_encoding_dims
        if self.time_encoding_dims:
            self.time_encoder = TimeEncoder(time_encoding_dims)
        self.datetime_encoder = DateTimeEncoder()

    def get_class_weights(self, count_type):
        '''
        Method to get weights for each label class, on the premise that we balance the dataset out.
        count_type : 'label_segment_count' - balances the data segments , 'label_point_count' - balances the datapoints
        '''

        if count_type != 'None':
            counts = self.metadata[count_type]
            weights = torch.ones(len(counts))
            for label, count in counts.items():
                weights[self.labels_map[label]] = 1/count
        else:
            counts = self.metadata['label_segment_count']
            weights = torch.ones(len(counts))

        weights = weights/torch.min(weights)
        return weights

    def __len__(self):
        '''
        Get the length of dataset.
        '''
        return len(self.metadata['track_ids'])

    def __getitem__(self, index):
        '''
        Get the item for each batch
        :return: a tuple of 6 object:
        1) normalized features of dataset
        2) labels of dataset (one-hot encoded and labels_dct)
        3) labels of dataset (encoded with labels_dct)
        4) length of each sequences without padding
        5) track id of each row in the dataset
        '''
        track_id = self.idx_to_track_id[index]
        data_slice = self.data.loc[track_id].compute().assign(
            time_from_start=lambda x: (x.datetime-x.datetime.min()).dt.total_seconds())

        x = torch.from_numpy(data_slice[self.features].values)
        x = self.transforms(x)

        if self.time_encoding_dims:
            x_time = torch.from_numpy(data_slice['time_from_start'].values)
            x_time_encoded = self.time_encoder(x_time)
            x = torch.cat([x, x_time_encoded], -1)

        x_datetime = self.datetime_encoder(data_slice['datetime'])
        x = torch.cat([x, x_datetime], -1)

        y = torch.from_numpy(data_slice[self.label].map(
            self.labels_map).astype(int).values)
        # labels, label_index = preprocess_y(self.data_labels[index], self.labels_dct, self.mode)

        callsign_string = np.array(
            data_slice[self.callsign_column].map(self._map_callsign).tolist())
        callsign_index = torch.from_numpy(callsign_string)

        mode3_string = np.array(
            data_slice[self.mode3_column].map(self._map_mode3).tolist())
        mode3_index = torch.from_numpy(mode3_string)

        # make sure that the sequence length is the last to be returned for batch collation
        return x, y, mode3_index, callsign_index, self.length_mapper[track_id]
