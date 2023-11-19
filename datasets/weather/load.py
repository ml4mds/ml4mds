# -*- coding: utf-8 -*-
"""Data loader class for Weather dataset."""

import numpy as np
import pandas as pd


class WeatherStreams:
    """Weather dataset.

    10 data streams, 8 features and 49728 samples.
    """

    def __init__(self):
        """__init__ for Weather."""
        df = pd.read_csv('datasets/weather/data.csv')
        ser_label = df.pop('w10m_obs')
        self.x = np.zeros((49728, 10, 8))
        self.y = np.zeros((49728, 10))
        for i in range(10):
            i1 = i * 49728
            i2 = i1 + 49728
            self.x[:, i, :] = df.iloc[i1:i2, :].to_numpy()
            self.y[:, i] = ser_label.iloc[i1:i2].to_numpy()
        del df

    def __len__(self):
        """Methods for emulating a container type."""
        return 49728

    def __getitem__(self, key):
        """Methods for emulating a container type."""
        return self.x[key], self.y[key]
