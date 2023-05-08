# -*- coding: utf-8 -*-
"""Data loader.

Data loaders, all datasets are indexed in size of (t, m, d),
where t is the timestamp of data,
m is the number of data streams,
and d is the dimension of the feature.
"""
import numpy as np
import pandas as pd
import streamlit as st


class TrainStreams:
    """SydneyTrain dataset."""

    def __init__(self):
        """__init__ for TrainStreams."""
        df = pd.read_csv('data/multistream.central.csv')
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


class WeatherStreams:
    """Weather dataset."""

    def __init__(self):
        """__init__ for Weather."""
        df = pd.read_csv('data/multistream.weather.csv')
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


class SensorStreams:
    """Sensor dataset."""

    def __init__(self):
        """__init__ for SensorStreams."""
        df = pd.read_csv('data/multistream.sensor.csv')
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


@st.cache_data
def loaddata(dataset):
    """Load data."""
    if dataset == "Train":
        ss = TrainStreams()
    elif dataset == "Weather":
        ss = WeatherStreams()
    elif dataset == "Sensor":
        ss = SensorStreams()
    else:
        st.write("Error: Please select a correct dataset!")
    return ss


if __name__ == "__main__":
    print('Test SydneyTrain dataset...')
    train = TrainStreams()
    for i in range(10):
        x, y = train[i:i+8]
        print(x.shape, y.shape)
    print('Test complete!')
    print()
    print('Test Weather dataset...')
    weather = WeatherStreams()
    for i in range(10):
        x, y = weather[i:i+8]
        print(x.shape, y.shape)
    print('Test complete!')
    print()
    print('Test Sensor dataset...')
    sensor = SensorStreams()
    for i in range(10):
        x, y = sensor[i:i+8]
        print(x.shape, y.shape)
    print('Test complete!')
