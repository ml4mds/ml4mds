"""The dynamic multi-stream handler interface."""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

from utils.dataloader import loaddata
from utils.dataloader import load_example
from utils.dataloader import load_visdata
from utils.functions import stream_selector
from algorithms.baseline.model import BaselineMultiStreamHandler
from algorithms.fuzzm.model import FuzzmddMultiStreamHandler
from algorithms.fuzzm.model import FuzzmdaMultiStreamHandler

# page config
st.set_page_config(page_title="Machine Learning for Multiple Data Streams",
                   layout="wide")
if 'flag' not in st.session_state:
    st.session_state.flag = False

###############################
#            title            #
###############################
st.title("Machine Learning for :red[Multiple] Data Streams")
subtitle, buttons = st.columns(2, gap="large")
subtitle.caption("Handling machine learning problems for multiple data streams"
                 " with unpredicted concept drifts.")
button_placeholders = buttons.columns(9)
button_run = button_placeholders[-2].button("Run", type="primary")
button_reset = button_placeholders[-1].button("Reset", type="primary")
if button_reset:
    st.session_state.flag = False
if button_run:
    st.session_state.flag = True

placeholder = st.empty()
########################################
#            before running            #
########################################
with placeholder.container():
    # ========== datasets & algorithms ==========
    tab_data, tab_algo = st.tabs(["Datasets", "Algorithms"])
    # datasets
    col_data1, col_data2 = tab_data.columns(2)
    dataset = col_data1.selectbox(
            "Select a dataset or upload your own dataset:",
            ("Upload your own dataset", "Weather", "Train", "Sensor"))
    streams_selection = col_data1.text_input(
            "Select all or some of data streams "
            "(e.g.: 'all', '1, 3', '1-3' or '1-3,5')",
            "all")
    if dataset == "Upload your own dataset":
        col_data1.file_uploader("CSV support only", type="csv")
    else:
        df = load_example(dataset)
        col_data1.dataframe(df)
        vals = load_visdata(dataset)
        m, n = vals.shape
        vis_percent = col_data2.slider("How much data"
                                       " would you like to display (%)?",
                                       0.0, 100.0, 100.0, 0.01)
        vals1 = vals[:, :int(n*vis_percent/100)]
        vis_tabs = col_data2.tabs(["Streams #{}".format(i+1) for i in range(m)])
        for i in range(m):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title("Visualization on Stream #{}".format(i+1))
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.plot(vals1[i, :], '.')
            fig.subplots_adjust(top=1, bottom=0, left=0, right=1.6, hspace=0.3)
            vis_tabs[i].pyplot(fig)
    # algorithms
    col_algo1, col_algo2 = tab_algo.columns(2)
    method = col_algo1.selectbox("Select algorithms:",
                                 ("None", "FuzzMDD + FuzzMDA"))
    training_size = col_algo2.number_input("Training set size (>0): ", value=100)
    batch_size = col_algo2.number_input("Batch size (>0): ", value=100)
    eval_method = col_algo2.selectbox("Evaluate the accuray via:",
                                 ("Mean squared error (MSE)",
                                 "Mean absolute error (MAE)",
                                 "Root mean squared error (RMSE)"))

#######################################
#            after running            #
#######################################
if st.session_state.flag:
    with placeholder.container():
        # ========== loading data ==========
        ss = loaddata(dataset)
        n, m, d = ss.x.shape
        # ========== initialize models ==========
        x_train, y_train = ss[:training_size]
        # baseline
        baseline = BaselineMultiStreamHandler(m, linear_model.Ridge(alpha=1))
        baseline.fit(x_train, y_train)
        # method to be evaulated
        method1 = None
        method2 = None
        if method == "FuzzMDD + FuzzMDA":
            method1 = FuzzmddMultiStreamHandler(m, linear_model.Ridge(alpha=1))
            method2 = FuzzmdaMultiStreamHandler(m, linear_model.Ridge(alpha=1))
        method1.fit(x_train, y_train)
        method2.fit(x_train, y_train)
        # ========== simulation starts ==========
        bar = st.progress(0)
        if streams_selection == "all":
            streams = list(range(m))
        else:
            streams = stream_selector(streams_selection)
        monitor_tabs = st.tabs(["Streams #{}".format(i+1) for i in streams])
        figures_placeholders = [tab.empty() for tab in monitor_tabs]
        N = (n - training_size) // batch_size
        window_size = 30
        timestamp = np.arange(1, N+1) * batch_size
        results = np.zeros((3, m, N))
        results_avg = np.zeros((3, m, N))
        results_drift = np.zeros((2, m, N))
        for i in range(N):
            i1 = training_size + i * batch_size
            i2 = i1 + batch_size
            x, y = ss[i1:i2]
            # baseline
            results[0, :, i] = baseline.score(x, y)
            if i == 0:
                results_avg[0, :, 0] = results[0, :, 0]
            else:
                results_avg[0, :, i] = \
                        (results_avg[0, :, i-1]*i + results[0, :, i]) / (i + 1)

            # method to be evaulated
            results[1, :, i], dlist1 = method1.score(x, y)
            if i == 0:
                results_avg[1, :, 0] = results[1, :, 0]
            else:
                results_avg[1, :, i] = \
                        (results_avg[1, :, i-1]*i + results[1, :, i]) / (i + 1)
            results_drift[0, dlist1, i] = results[1, dlist1, i]
            results[2, :, i], dlist2 = method2.score(x, y)
            if i == 0:
                results_avg[2, :, 0] = results[2, :, 0]
            else:
                results_avg[2, :, i] = \
                        (results_avg[2, :, i-1]*i + results[2, :, i]) / (i + 1)
            results_drift[1, dlist2, i] = results[2, dlist2, i]

            bar.progress((i+1)/N)
            if i + 1 < window_size:
                il = 0
            else:
                il = i + 1 - window_size
            if i < 10:
                legend_loc = 1
            else:
                legend_loc = 2
            for j, stream_j in enumerate(streams):
                fig = plt.figure()
                ax1 = fig.add_subplot(2, 1, 1)
                ax1.set_title("Model performance:"
                              " of Stream #{}".format(stream_j+1))
                ax1.set_ylabel(eval_method)
                ax1.set_xlim(timestamp[il], timestamp[il+window_size])
                ax1.plot(timestamp[il:i+1], results[0, stream_j, il:i+1],
                         color="#0095ff",
                         label="w/o adaptation")
                ax1.plot(timestamp[il:i+1], results[1, stream_j, il:i+1],
                         color="#00aa3c",
                         label="adaptation w/o correlations")
                ax1.plot(timestamp[il:i+1], results[2, stream_j, il:i+1],
                         color="#ff4841",
                         label="adaptation with correlations")
                ax1.plot(timestamp[il:i+1], results_avg[0, stream_j, il:i+1], ':',
                         color="#0095ff",
                         label="w/o adaptation (avg)")
                ax1.plot(timestamp[il:i+1], results_avg[1, stream_j, il:i+1], ':',
                         color="#00aa3c",
                         label="adaptation w/o correlations (avg)")
                ax1.plot(timestamp[il:i+1], results_avg[2, stream_j, il:i+1], ':',
                         color="#ff4841",
                         label="adaptation with correlations (avg)")
                drift_points = results_drift[0, stream_j, il:il+window_size] > 0
                ax1.plot(timestamp[il:il+window_size][drift_points],
                         results_drift[0, stream_j, il:il+window_size][drift_points], '^',
                         color="#00aa3c")
                drift_points = results_drift[1, stream_j, il:il+window_size] > 0
                ax1.plot(timestamp[il:il+window_size][drift_points],
                         results_drift[1, stream_j, il:il+window_size][drift_points], '^',
                         color="#ff4841")
                ax1.legend(loc=legend_loc)
                ax2 = fig.add_subplot(2, 1, 2)
                ax2.set_title("Visualization of Stream #{}".format(stream_j+1))
                ax2.set_xlabel("Time")
                ax2.set_ylabel("Value")
                ax2.set_xlim(timestamp[il], timestamp[il+window_size])
                ax2.plot(np.arange(timestamp[il], timestamp[i]+1),
                         vals[stream_j, training_size+timestamp[il]:training_size+timestamp[i]+1], '.')
                fig.subplots_adjust(top=1, bottom=0,
                                    left=0, right=2.5, hspace=0.3)
                figures_placeholders[j].pyplot(fig)
