import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, Flatten, BatchNormalization
from sklearn.metrics import mean_squared_error, r2_score