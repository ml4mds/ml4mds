# -*- coding: utf-8 -*-
"""Data loader class for Sensor dataset."""

import numpy as np
import pandas as pd


class SensorStreams:
    """Sensor dataset.

    6 data streams, 3 features and 20844 samples.
    """

    def __init__(self):
        """__init__ for SensorStreams."""
        df = pd.read_csv('datasets/sensor/sensor.csv')
        self.x = np.zeros((20844, 6, 3))
        self.y = np.zeros((20844, 6))
        for i in range(6):
            i1 = 4 * i
            i2 = i1 + 3
            self.x[:, i, :] = df.iloc[:, i1:i2].to_numpy()
            self.y[:, i] = df.iloc[:, i2].to_numpy()
        del df

    def __len__(self):
        """Methods for emulating a container type."""
        return 20844

    def __getitem__(self, key):
        """Methods for emulating a container type."""
        return self.x[key], self.y[key]
