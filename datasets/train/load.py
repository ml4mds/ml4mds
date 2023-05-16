# -*- coding: utf-8 -*-
"""Data loader class for Train dataset."""

import numpy as np
import pandas as pd


class TrainStreams:
    """SydneyTrain dataset.

    8 data streams, 11 features and 373327 samples.
    """

    def __init__(self):
        """__init__ for TrainStreams."""
        df = pd.read_csv('datasets/train/train.csv')
        self.x = np.zeros((373327, 8, 11))
        self.y = np.zeros((373327, 8))
        for i in range(8):
            self.x[:, i, 0] = df['PERIODS'].to_numpy()
            self.x[:, i, 1:3] = df[['TRIP_ORGIN_STATION',
                                    'TRIP_DESTINATION_STATION']].to_numpy()
            self.x[:, i, 3] = df['PLANNED_ARRIVAL_PLATFORM'].to_numpy()
            self.x[:, i, 4:] = df[['SEGMENT_DIRECTION_NAME',
                                   'TRIP_DURATION_SEC', 'DWELL_TIME',
                                   'DAY_OF_YEAR', 'DAY_OF_WEEK', 'TIME',
                                   'CAR{}_PSNGLD_ARRIVE'.format(i+1)
                                   ]].to_numpy()
            self.y[:, i] = df['CAR{}_PSNGLD_DEPART'.format(i+1)].to_numpy()
        del df

    def __len__(self):
        """Methods for emulating a container type."""
        return 373327

    def __getitem__(self, key):
        """Methods for emulating a container type."""
        return self.x[key], self.y[key]
