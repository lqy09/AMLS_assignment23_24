import numpy as np

data = np.load('./Datasets/pneumoniamnist.npz')
print(data)

for file in data.files:
    print(file)

data.close()
