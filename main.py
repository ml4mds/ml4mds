"""My Streamlit app."""

import streamlit as st
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

from loader import TrainStreams
from loader import WeatherStreams
from algorithms.fuzzm.model import MultiStreamRetrainer
from algorithms.fuzzm.model import MultiStreamHandler

st.set_page_config(layout="wide")

# sidebar:
datasets = {
        "None": "",
        "Train": "The Train dataset records train dwelling information"
        "There are eight data streams in the data set."
        "Each data stream is corresponding to a train station."
        "The task is to predict the load change of carriages"
        "during the train dwelling time,"
        "so that staff can organize the crowd and avoid congestion",
        "Weather": "The Weather data set records weather data"
        "from ten weather stations near Beijing China."
        "There are ten data streams in the data set."
        "Each data stream is corresponding to a weather station."
        "The weather data includes temperature, pressure, humidity, etc.."
        "The task is to predict the accumulated rainfall in one hour."
        }


@st.cache
def loaddata(dataset):
    """Load data."""
    if dataset == "Train":
        ss = TrainStreams()
    elif dataset == "Weather":
        ss = WeatherStreams()
    else:
        st.write("Error: Please select a correct dataset!")
    return ss


st.sidebar.header("Datasets")
dataset = st.sidebar.selectbox("Select a Dataset:",
                               ("None", "Weather"))
st.sidebar.write(datasets[dataset])

st.sidebar.header("Algorithms")
st.sidebar.write("Select Algorithms:")
method1 = st.sidebar.checkbox("FuzzMDD + Re-training")
method2 = st.sidebar.checkbox("FuzzMDD + FuzzMDA")

st.sidebar.header("Streams setting")
training_size = st.sidebar.number_input("training set size: ", value=100)
batch_size = st.sidebar.number_input("batch size: ", value=100)

if st.sidebar.button("Run", type="primary"):
    ss = loaddata(dataset)
    n, m, d = ss.x.shape
    x_train, y_train = ss[:training_size]
    result = np.zeros((2, m, n))
    # ========== initialize models ==========
    # FuzzMDD + Re-training
    retrain = None
    if method1:
        retrain = MultiStreamRetrainer(m, linear_model.Ridge(alpha=1))
        retrain.fit(x_train, y_train)
    # FuzzMDD + FuzzMDA
    fmda = None
    if method2:
        fmda = MultiStreamHandler(m, linear_model.Ridge(alpha=1))
        fmda.fit(x_train, y_train)

    # ========== simulation starts ==========
    bar = st.progress(0)
    placeholder = st.empty()
    N = (n - training_size) // batch_size
    for i in range(N):
        i1 = training_size + i * batch_size
        i2 = i1 + batch_size
        x, y = ss[i1:i2]
        # retrain
        # FuzzMDD + Re-training
        if method1:
            result[0, :, i], dlist1 = retrain.score(x, y)
        # FuzzMDD + FuzzMDA
        if method2:
            result[1, :, i], dlist2 = fmda.score(x, y)

        bar.progress(i)
        with placeholder.container():
            fig = plt.figure()
            for j in range(m):
                ax = fig.add_subplot(m//3+1, 3, j+1)
                ax.set_title("Stream #{}".format(j+1))
                ax.set_xlabel("Batch #")
                ax.set_ylabel("MSE")
                if method1:
                    ax.plot(result[0, j, :i+1], label="FuzzMDD + Retraining")
                    if j in dlist1:
                        ax.plot([i], result[0, j, i], "r*")
                if method2:
                    ax.plot(result[1, j, :i+1], label="FuzzMDD + FuzzMDA")
                    if j in dlist2:
                        ax.plot([i], result[1, j, i], "r*")
                ax.legend()
            fig.subplots_adjust(top=m//3+1, bottom=0,
                                left=0, right=3, hspace=0.3)
            st.pyplot(fig)
