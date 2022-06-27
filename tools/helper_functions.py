import numpy as np


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