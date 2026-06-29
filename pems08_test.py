import numpy as np

data = np.load(r"C:\Users\student\paperTest\pems08.npz")

print(data.files)

for k in data.files:
    print(k, data[k].shape)