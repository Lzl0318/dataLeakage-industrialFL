import numpy as np

train_data = np.load('./original/train_data.npy')
test_data = np.load('./original/test_data.npy')
train_label = np.load('./original/train_label.npy')
test_label = np.load('./original/test_label.npy')

# train_data_classify = np.hstack((train_data[:, 0:22], train_data[:, 41:]))
# test_data_classify = np.hstack((test_data[:, 0:22], test_data[:, 41:]))
#
train_data_regression = np.hstack((train_data[0:600, 0:22], train_data[0:600:, 41:]))
test_data_regression = np.hstack((test_data[0:200, 0:22], test_data[0:200, 41:]))
#
train_label_regression = train_data[0:600, [24, 25, 27, 29, 30, 35]]
test_label_regression = test_data[0:200, [24, 25, 27, 29, 30, 35]]
#
# np.save('./classify/train_data.npy', train_data_classify)
# np.save('./classify/test_data.npy', test_data_classify)
#
np.save('./regression/train_data.npy', train_data_regression)
np.save('./regression/test_data.npy', test_data_regression)
#
np.save('./regression/train_label.npy', train_label_regression)
np.save('./regression/test_label.npy', test_label_regression)

# train_data_median = np.zeros((22, 33))
# for i in range(22):
#     train_data_median[i, :] = np.median(train_data[0+i*600:600+i*600], axis=0)
# np.save('./classify/train_data_median.npy', train_data_median)

