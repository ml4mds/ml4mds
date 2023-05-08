"""The dynamic multi-stream handler interface."""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from utils.dataloader import loaddata
from utils.functions import stream_selector
from algorithms.fuzzm.model import MultiStreamHandler

# page config
st.set_page_config(page_title="Machine Learning for Multiple Data Streams",
                   layout="wide")

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
    button_run = False
if button_run:
    button_reset = False
st.divider()

########################################
#            before running            #
########################################
placeholder = st.empty()
with placeholder.container():
    # ========== datasets & algorithms ==========
    col_data, col_algo = st.columns(2, gap="large")
    # datasets
    col_data.header("Datasets")
    col_data1, col_data2 = col_data.columns(2)
    dataset = col_data1.selectbox("Select a dataset or upload your own dataset:",
                                   ("Upload your own dataset", "Weather", "Train", "Sensor"))
    streams_selection = col_data1.text_input("Select all or some of data streams "
                                         "(e.g.: 'all', '1, 3', '1-3' or '1-3,5')",
                                         "all")
    if dataset == "Upload your own dataset":
        col_data2.file_uploader("CSV support only", type="csv")
    elif dataset == "Weather":
        df = pd.read_csv('data/multistream.weather.csv')
        col_data2.dataframe(df.head())
        del df
    elif dataset == "Train":
        df = pd.read_csv('data/multistream.central.csv')
        col_data2.dataframe(df.head())
        del df
    elif dataset == "Sensor":
        df = pd.read_csv('data/multistream.sensor.csv')
        col_data2.dataframe(df.head())
        del df
    # algorithms
    col_algo.header("Algorithms")
    col_algo1, col_algo2 = col_algo.columns(2)
    method = col_algo1.selectbox("Select algorithms:",
                                 ("None", "FuzzMDD + FuzzMDA"))
    # ========== other parameters ==========
    other_paras = st.columns(5, gap="large")
    training_size = other_paras[0].number_input("Training set size (>0): ", value=100)
    batch_size = other_paras[1].number_input("Batch size (>0): ", value=100)
    eval_method = other_paras[2].selectbox("Evaluate the accuray via:",
                                 ("Mean squared error (MSE)",
                                 "Mean absolute error (MAE)",
                                 "Root mean squared error (RMSE)"))

#######################################
#            after running            #
#######################################
if button_run:
    with placeholder.container():
        # ========== loading data ==========
        ss = loaddata(dataset)
        n, m, d = ss.x.shape
        # ========== initialize models ==========
        x_train, y_train = ss[:training_size]
        pca = PCA(n_components=1)
        pca.fit(np.hstack((x_train.reshape((-1, d)),
                           y_train.reshape((-1, 1)))))
        # FuzzMDD + FuzzMDA
        fmda = None
        if method == "FuzzMDD + FuzzMDA":
            fmda = MultiStreamHandler(m, linear_model.Ridge(alpha=1))
            fmda.fit(x_train, y_train)
        # ========== simulation starts ==========
        bar = st.progress(0)
        N = (n - training_size) // batch_size
        vis_data = np.zeros((m, N))
        results = np.zeros((m, N))
        fig_plotting = st.empty()
        for i in range(N):
            i1 = training_size + i * batch_size
            i2 = i1 + batch_size
            x, y = ss[i1:i2]
            # FuzzMDD + FuzzMDA
            if method == "FuzzMDD + FuzzMDA":
                results[:, i], dlist1 = fmda.score(x, y)

            bar.progress((i+1)/N)
            fig = plt.figure()
            if streams_selection == "all":
                streams = list(range(m))
            else:
                streams = stream_selector(streams_selection)
            for j, stream_j in enumerate(streams):
                temp = pca.transform(np.hstack((x[:, stream_j, :],
                                                y[:, stream_j].reshape(-1, 1))))
                temp = temp.reshape(-1)
                vis_data[stream_j, i] = temp.mean()
                ax1 = fig.add_subplot(len(streams), 2, 2*j+1)
                ax1.set_title("Visualization on Stream #{}".format(stream_j+1))
                ax1.set_xlabel("Batch #")
                ax1.set_ylabel("Data")
                ax1.plot(vis_data[stream_j, :i+1], "ro-")

                ax2 = fig.add_subplot(len(streams), 2, 2*j+2)
                ax2.set_title("Accuracy on Stream #{}".format(stream_j+1))
                ax2.set_xlabel("Batch #")
                ax2.set_ylabel(eval_method)
                if method == "FuzzMDD + FuzzMDA":
                    ax2.plot(results[stream_j, :i+1], label="FuzzMDD + FuzzMDA")
                    if stream_j in dlist1:
                        ax2.text(0.5, 0.5, "Drift detected!", color="red",
                                 fontweight="bold")
                ax2.legend()
            fig.subplots_adjust(top=len(streams), bottom=0,
                                left=0, right=2, hspace=0.3)
            if i >= 2:
                fig_plotting.pyplot(fig)
