import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy import spatial
import argparse


# 超参数
parser = argparse.ArgumentParser(description='dlg for TE.')
parser.add_argument('--lr', type=float, default=0.004, help='the learning rate of attack')
parser.add_argument('--classes', type=int, default=21, help='the class of dataset')
parser.add_argument('--protection', type=int, default=1, help='whether protection or not')
args = parser.parse_args()


# define net
class MLP(nn.Module):
    def __init__(self, n_input, n_output):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(n_input, 512)
        self.hidden2 = nn.Linear(512, 256)
        self.hidden3 = nn.Linear(256, 128)
        self.predict = nn.Linear(128, n_output)

    def forward(self, x):
        out = self.hidden1(x)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.relu(out)
        out = self.hidden3(out)
        out = F.relu(out)
        out = self.predict(out)

        return out


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


def label_to_onehot(target, num_classes=22):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def gradient_compression(tensor_list):
    new_tensor_list = []
    for tensor in tensor_list:
        array = tensor.numpy().flatten()
        # threshold = np.sort(np.absolute(array))[int(array.shape[0]*value)]
        threshold = 0.005
        array = np.where(np.absolute(array) > threshold, array, 0)
        new_tensor = torch.tensor(array.reshape((tensor.shape[0], -1))).squeeze()
        new_tensor_list.append(new_tensor)

    return new_tensor_list


# 加载模型
model = MLP(33, 22)
torch.manual_seed(1234)
model.apply(weights_init)


# 此时新的数据被送入模型,攻击开始
data = torch.from_numpy(np.load('../data/TE/classify/train_data_median.npy')[args.classes]).float().unsqueeze(0)
label = torch.tensor([args.classes]).float().type(torch.LongTensor)
one_hot_label = label_to_onehot(label, 22)

# 损失函数
criterion = cross_entropy_for_onehot

# 计算原始梯度
pred = model(data)
y = criterion(pred, one_hot_label)
dy_dx = torch.autograd.grad(y, model.parameters())
original_dy_dx = list((_.detach().clone() for _ in dy_dx))
if args.protection == 1:
    original_dy_dx = gradient_compression(original_dy_dx)

# generate dummy data and label
dummy_data = torch.randn(data.size()).requires_grad_(True)
dummy_label = torch.randn(one_hot_label.size()).requires_grad_(True)


# dummy_label = ((torch.zeros(one_hot_label.size())+torch.ones(one_hot_label.size()))/2).requires_grad_(True)

# 优化器
optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=args.lr)

# begin loop
for iters in range(300):
    def closure():
        optimizer.zero_grad()
        dummy_pred = model(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()

        return grad_diff

    optimizer.step(closure)

    if iters % 10 == 0:
        current_loss = closure()
        print(iters, "gradient: %.4f" % current_loss.item())


data = data.squeeze().numpy()
dummy_data = dummy_data.detach().squeeze().numpy()

if args.protection == 1:
    np.save('./results/dlg/reverse_result_protected/' + str(args.classes) + '.npy', dummy_data)
else:
    np.save('./results/dlg/reverse_result/' + str(args.classes) + '.npy', dummy_data)

mse = mean_squared_error(data, dummy_data)
cos = 1 - spatial.distance.cosine(data, dummy_data)

print(data)
print(dummy_data)
print('\nThe mean_squared_error is {}'.format(mse))
print('\nThe cosine_similarity is {}'.format(cos))







