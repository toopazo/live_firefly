import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def split_sequences(index_array, min_length=0):
    """ splits the index array into sequences. Discards sequences with length below min_length"""

    sequences = []

    ptr1 = 0
    ptr2 = 0

    while ptr1 < index_array.size:

        sequence = []

        while ptr1 < index_array.size and index_array[ptr1] == ptr2:
            sequence.append(index_array[ptr1])
            ptr1 += 1
            ptr2 += 1

        if sequence and len(sequence) >= min_length:
            sequences.append(sequence)

        ptr2 += 1

    return sequences


def normalize(data_vector, method='zscore'):
    """ Normalize input vector with either minmax or zscore transformation"""

    if method == 'minmax':
        minimum = np.min(data_vector)
        maximum = np.max(data_vector)

        return (data_vector - minimum) / (maximum - minimum)

    if method == 'zscore':
        return (data_vector - np.mean(data_vector)) / np.std(data_vector)

    if method == '':  # do nothing

        return data_vector


def rolling_normal(data, window=60, method='minmax'):

    if method == 'minmax':
        minimum = data.rolling(window, 1).min()
        maximum = data.rolling(window, 1).max()

        normData = (data - minimum) / (maximum - minimum)

    else:
        rollingMean = data.rolling(window, 1).mean()
        rollingStd = data.rolling(window, 1).std()

        normData = (data - rollingMean) / rollingStd

    if normData.isna().iloc[0].sum() > 0:
        normData.iloc[0] = normData.iloc[1]

    return normData


def moving_average(in_data, window=5):
    """ Calculates the moving average of the values for every dictionary"""

    # get values of DataFrame
    if type(in_data) is pd.Series:
        in_data = in_data.values
        return np.convolve(in_data, np.ones(window), 'same') / window

    # do moving average for numpy array
    if type(in_data) is np.ndarray:
        return np.convolve(in_data, np.ones(window), 'same') / window


def get_y_bound(y_values, t0, t1):
    """ Get the upper and lower y boundaries for power plot """
    minimum = 300
    maximum = 0

    if type(y_values) is dict:

        for array in list(y_values.values()):
            minimum = min(minimum, np.min(array[t0:t1]))
            maximum = max(maximum, np.max(array[t0:t1]))

    else:
        minimum = np.min(y_values[t0:t1])
        maximum = np.max(y_values[t0:t1])

    return [minimum, maximum]


def perform_polynomial_regression(data, x, y):
    # regression for power vs. dRPM

    X = data[[x]]
    Y = data[[y]]

    regressionRange = np.linspace(X.min(), X.max(), 1000)

    features = PolynomialFeatures(degree=2, include_bias=True).fit_transform(X)
    dRpmRegression = LinearRegression(fit_intercept=False, positive=False)
    dRpmRegression.fit(features, Y)

    dRpmRangePoly = PolynomialFeatures(degree=2, include_bias=True).fit_transform(regressionRange)
    regressionCurve = dRpmRegression.predict(dRpmRangePoly)

    return regressionRange, regressionCurve