import numpy as np
inputs = [[2.0, 3.0, 4.0, 5.0],
          [2.1, 3.2, 4.3, 5.4],
          [2.2, 3.3, 4.4, 5.5],
          [2.3, 3.4, 4.5, 5.6],
          [2.4, 3.5, 4.6, 5.7]]
weights = [[1.2, 2.8, -3.5, -4.1],
           [2.1, 3.2, -4.3, -5.4],
           [2.0, 3.0, -4.0, -5.5],
           [2.2, 3.3, -4.4, -5.6],
           [2.3, 3.4, -4.5, -5.7]]
biases = [2.0, 3.0, 4.2, 3.1, 3.3]

layers_output = np.dot(inputs, np.array(weights).T) + biases

weights2 = [[0.5, -1.0, 1.5, 2.3, 5.1],
            [1.0, -0.5, -1.5, 3.1, 3.7],
            [5.2, 3.6, 7.1, 5.5, 6.1]]
biases2 = [0.5, -0.5, 5.2]

layers_output1 = np.dot(layers_output, np.array(weights2).T) + biases2


print(layers_output1)