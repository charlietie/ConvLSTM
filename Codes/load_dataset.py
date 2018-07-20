import numpy as np
import pickle as cp



def load_dataset(filename):
    '''

    :param filename:
    :return:
              X_train
              y_train
              X_test
              y_test
    '''

    """
    Function to load dataset

    Argument:
        filename : the path of the preprocessed dataset

    Return:

        X_train
        y_train
        X_test
        y_test
    Notice:
        Need use `pickle(python3)` or`cpicckle(python2)` module to load and read dataset.


    """

    f = open(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: X_train {0}, X_test {1}".format(
        X_train.shape, X_test.shape))
    print(" ..reading instances: y_train {0}, y_test {1}".format(
        y_train.shape, y_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test