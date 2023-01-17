import numpy as np
for i in range(22):
    data = np.load('./'+str(i)+'.npy')
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    np.save('./'+str(i)+'.npy', data)
