# DATA

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time, os, requests, zipfile
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# LSTM_returns

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, Flatten, BatchNormalization
from sklearn.metrics import mean_squared_error, r2_score


# optimization

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt