import re

import torch
from torchvision.transforms import Compose
import numpy as np

from . import transforms
from .config import cfg as model_cfg


class PreProcessor():

    CALLSIGN_REGEX_FILTER = '[^a-z0-9]'
    MODE3_REGEX_FILTER = '[^0-9]'
    CALLSIGN_CHARACTER_LIMIT = 7
    MODE3_CHARACTER_LIMIT = 4
    UNKNOWN_TOKEN = '-'
    PAD_TOKEN = '_'
    unknown_callsign = UNKNOWN_TOKEN*CALLSIGN_CHARACTER_LIMIT
    unknown_mode3 = UNKNOWN_TOKEN*MODE3_CHARACTER_LIMIT

    # note that _ is padding to reach charlimit and -  is unknown
    CALLSIGN_TOKENS = '0123456789abcdefghijklmnopqrstuvwxyz-_'
    CALLSIGN_CHAR2IDX = {v: i for i, v in enumerate(CALLSIGN_TOKENS)}

    # note that _ is padding to reach charlimit and -  is unknown
    MODE3_TOKENS = '0123456789-_'
    MODE3_CHAR2IDX = {v: i for i, v in enumerate(MODE3_TOKENS)}

    def __init__(self):

        super().__init__()

        transforms_list = transforms.get_transforms(
            model_cfg['data']['transforms'])
        self.transforms = Compose(transforms_list)
        self.time_encoder = transforms.TimeEncoder(
            model_cfg['data']['time_encoding_dims'])
        self.datetime_encoder = transforms.DateTimeEncoder()

        # with open(os.path.join(os.path.dirname(
        #         os.path.abspath(__file__)), 'metadata.json')) as infile:
        #     self.metadata = json.load(infile)
        # self.labels_map = {l: i for i, l in enumerate(self.metadata['labels'])}

        # self.n_classes = len(self.metadata['labels'])
        # self.labels_map['start'] = len(self.metadata['labels'])
        # self.labels_map['pad'] = len(self.metadata['labels']) + 1

        # self.inverse_labels_map = {v: k for k, v in self.labels_map.items()}

        self.mode3_padding_value = self.CALLSIGN_CHAR2IDX['_']
        self.callsign_padding_value = self.MODE3_CHAR2IDX['_']
        # self.get_label_string = np.vectorize(self.inverse_labels_map.get)

    def _map_callsign(self, string):
        return [self.CALLSIGN_CHAR2IDX[i] for i in string]

    def _map_mode3(self, string):
        return [self.MODE3_CHAR2IDX[i] for i in string]

    def clean_callsign(self, string):

        string = string.lower()

        if string == '':
            return self.unknown_callsign

        # special cases because of old code errors

        string = re.sub(self.CALLSIGN_REGEX_FILTER, '', string)
        if len(string) < self.CALLSIGN_CHARACTER_LIMIT:
            return string.ljust(self.CALLSIGN_CHARACTER_LIMIT, self.PAD_TOKEN)
        return string[0:self.CALLSIGN_CHARACTER_LIMIT]

    def clean_mode3(self, string):

        if string == '':
            return self.unknown_mode3

        string = re.sub(self.MODE3_REGEX_FILTER, '', string)
        if len(string) < self.MODE3_CHARACTER_LIMIT:
            return string.ljust(self.MODE3_CHARACTER_LIMIT, self.PAD_TOKEN)
        return string[0:self.MODE3_CHARACTER_LIMIT]

    def __call__(self, data_slice):

        x = torch.from_numpy(data_slice[model_cfg['data']['features']].values)
        x = self.transforms(x)

        if model_cfg['data']['time_encoding_dims']:
            x_time = torch.from_numpy(data_slice['time_from_start'].values)
            x_time_encoded = self.time_encoder(x_time)
            x = torch.cat([x, x_time_encoded], -1)

        x_datetime = self.datetime_encoder(data_slice['datetime'])
        x = torch.cat([x, x_datetime], -1)

        # y = torch.from_numpy(data_slice[self.label].map(
        #     self.labels_map).astype(int).values)
        # labels, label_index = preprocess_y(self.data_labels[index], self.labels_dct, self.mode)

        callsign_index = np.array(
            data_slice[model_cfg['data']['identifiers']['callsign_data_column']].map(self._map_callsign).tolist())
        callsign_index = torch.from_numpy(callsign_index)

        mode3_index = np.array(
            data_slice[model_cfg['data']['identifiers']['mode3_data_column']].map(self._map_mode3).tolist())
        mode3_index = torch.from_numpy(mode3_index)

        return x, mode3_index, callsign_index, data_slice['id'].values, x.shape[0]

    def collate(self, batch):
        # sort by len
        batch.sort(key=lambda x: x[-1], reverse=True)
        batch_x, batch_mode3, batch_callsign, batch_id, batch_len = zip(*batch)
        batch_pad_x = torch.nn.utils.rnn.pad_sequence(
            batch_x, batch_first=True).numpy()
        batch_pad_mode3 = torch.nn.utils.rnn.pad_sequence(
            batch_mode3, batch_first=True, padding_value=self.mode3_padding_value).numpy()
        batch_pad_callsign = torch.nn.utils.rnn.pad_sequence(
            batch_callsign, batch_first=True, padding_value=self.callsign_padding_value).numpy()

        return batch_pad_x, batch_pad_mode3, batch_pad_callsign, batch_len, batch_id
