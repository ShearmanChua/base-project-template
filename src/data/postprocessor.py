import json
import os

import numpy as np


class PostProcessor():

    def __init__(self):

        super().__init__()

        with open(os.path.join(os.path.dirname(
                os.path.abspath(__file__)), 'metadata.json')) as infile:
            self.metadata = json.load(infile)
        self.labels_map = {l: i for i, l in enumerate(self.metadata['labels'])}

        self.n_classes = len(self.metadata['labels'])
        self.labels_map['start'] = len(self.metadata['labels'])
        self.labels_map['pad'] = len(self.metadata['labels']) + 1

        self.inverse_labels_map = {v: k for k, v in self.labels_map.items()}
        self.y_padding_value = self.labels_map['pad']

        self.get_label_string = np.vectorize(self.inverse_labels_map.get)

    def __call__(self, predictions):

        predictions = self.get_label_string(predictions)

        return predictions
