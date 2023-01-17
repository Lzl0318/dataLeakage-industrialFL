import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 固定随机参数
setup_seed(47)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset
train_data = np.load('../data/TE/regression/train_data.npy')
test_data = np.load('../data/TE/regression/test_data.npy')
train_label = np.load('../data/TE/regression/train_label.npy')
test_label = np.load('../data/TE/regression/test_label.npy')

train_data = torch.from_numpy(train_data).float().to(device)
train_label = torch.from_numpy(train_label).float().to(device)
test_data = torch.from_numpy(test_data).float().to(device)
test_label = torch.from_numpy(test_label).float().to(device)

train_dataset = TensorDataset(train_data, train_label)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# dgc
def gradient_compression(tensor):
    array = tensor.numpy().flatten()
    # threshold = np.sort(np.absolute(array))[int(array.shape[0]*value)]
    threshold = 0.02
    array = np.where(np.absolute(array) > threshold, array, 0)
    new_tensor = torch.tensor(array.reshape((tensor.shape[0], -1))).squeeze()

    return new_tensor


# model
class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input, 512)
        self.hidden2 = nn.Linear(512, 256)
        self.hidden3 = nn.Linear(256, 128)
        self.predict = nn.Linear(128, n_output)

    def forward(self, x):
        out = self.hidden1(x)
        out = torch.relu(out)
        out = self.hidden2(out)
        out = torch.relu(out)
        out = self.hidden3(out)
        out = torch.relu(out)
        out = self.predict(out)

        return out


def train():
    model = Net(33, 6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    mselosslist = np.zeros(100)
    for i in range(100):
        for x, y in train_loader:
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            # 梯度稀疏化
            for j, param in enumerate(model.parameters()):
                if param.requires_grad:
                    param.grad.data = gradient_compression(param.grad.data)

            optimizer.step()

        out = model(test_data)
        mseloss = criterion(out, test_label)

        print('epoch = {}, mse = {}'.format(i, mseloss))
        mselosslist[i] = mseloss.item()

    # evaluation
    model.eval()
    pre = model(test_data)
    pred_y = pre.cpu().data.numpy().squeeze()
    target_y = test_label.cpu().data.numpy()
    for j in range(6):
        print('variable'+str(j)+':')
        print('mse_loss={:.4f}'.format(mean_squared_error(target_y[j], pred_y[j])))
        print('r2score={:.4f}\n'.format(r2_score(target_y[j], pred_y[j])))
    # torch.save(model.state_dict(), 'TE_regression_model.pt')
    # np.save('baseline_test_acc.npy', accarray)


if __name__ == "__main__":
    train()

















