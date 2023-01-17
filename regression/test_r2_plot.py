import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device: %s' % device)


# define net
# class MLP(nn.Module):
#     def __init__(self, n_input, n_output):
#         super(MLP, self).__init__()
#         self.hidden1 = nn.Linear(n_input, 512)
#         self.hidden2 = nn.Linear(512, 256)
#         self.hidden3 = nn.Linear(256, 128)
#         self.predict = nn.Linear(128, n_output)
#
#     def forward(self, x):
#         out = self.hidden1(x)
#         out = torch.relu(out)
#         out = self.hidden2(out)
#         out = torch.relu(out)
#         out = self.hidden3(out)
#         out = torch.relu(out)
#         out = self.predict(out)
#
#         return out
#
#
# model = MLP(33, 6).to(device)
# torch.manual_seed(1234)
# parameters = torch.load('./TE_regression_model.pt')
#
# weight4 = parameters['predict.weight'].cpu().numpy()
# weight4_new = weight4 + np.random.normal(0, 0.025, weight4.shape)
# parameters['predict.weight'] = torch.tensor(weight4_new).float().to(device)
#
# bias4 = parameters['predict.bias'].cpu().numpy()
# bias4_new = bias4 + np.random.normal(0, 0.025, bias4.shape)
# parameters['predict.bias'] = torch.tensor(bias4_new).float().to(device)
#
# model.load_state_dict(parameters)
#
# print('测试模型加噪声之后的回归性能')
# test_data = np.load('../data/TE/regression/test_data.npy')
# test_label = np.load('../data/TE/regression/test_label.npy')
# test_data = torch.from_numpy(test_data).float()
# test_label = torch.from_numpy(test_label).float()
#
# model.eval()
# model.cpu()
# pre = model(test_data)
# pred_y = pre.cpu().data.numpy().squeeze()
# target_y = test_label.cpu().data.numpy()
# for j in range(6):
#     print('variable'+str(j)+':')
#     print('mse_loss={:.4f}'.format(mean_squared_error(target_y[j], pred_y[j])))
#     print('r2score={:.4f}\n'.format(r2_score(target_y[j], pred_y[j])))

