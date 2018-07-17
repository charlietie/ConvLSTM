import numpy as np


def mini_batch(x, y, batch_size):
    mini_batches = []
    num_bathches = int(np.floor(x.shape[0] / batch_size))
    for batch_index in range(num_bathches):
        mini_batch_x = x[batch_index *
                         batch_size:((batch_index * batch_size) + batch_size)]
        mini_batch_y = y[batch_index *
                         batch_size:((batch_index * batch_size) + batch_size)]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)
    return mini_batches


x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
x
y = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
y

batch_size = 2

batches = mini_batch(x, y, batch_size)
batches
for batch in batches:
    x, y = batch
    print(x, "\n", y, "===\n")
