# -*- coding: utf-8 -*-
"""Data loading functions using st.cache_data."""

import numpy as np
import pandas as pd
import streamlit as st

from .train.load import TrainStreams
from .weather.load import WeatherStreams
from .sensor.load import SensorStreams


@st.cache_data
def load_data(dataset):
    """Load data."""
    ss = None
    if dataset == 'Train':
        ss = TrainStreams()
    elif dataset == 'Weather':
        ss = WeatherStreams()
    elif dataset == 'Sensor':
        ss = SensorStreams()
    return ss


@st.cache_data
def load_example(dataset):
    """Load examples."""
    file_path = 'datasets/' + dataset.lower() + '/example.csv'
    examples = pd.read_csv(file_path)
    return examples


@st.cache_data
def load_visdata(dataset, plot3d=False):
    """Load the data after dimension reuction."""
    if plot3d:
        suffix = '2d.npy'
    else:
        suffix = '.npy'
    file_path = 'datasets/' + dataset.lower() + '/visualization' + suffix
    with open(file_path, 'rb') as f:
        vals = np.load(f)
    return vals


if __name__ == "__main__":
    for dataset in ['Train', 'Weather', 'Sensor']:
        ss = load_data(dataset)
        print(ss.x.shape)
        print('Test load_data on ', dataset, ': success!')

        examples = load_example(dataset)
        print(examples)
        print('Test load_example on ', dataset, ': success!')

        vals = load_visdata(dataset)
        print(vals.shape)
        print('Test load_example on ', dataset, ': success!')

    print('========== Test completed! ==========')
