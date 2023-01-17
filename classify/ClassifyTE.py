import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, recall_score, precision_score


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
train_data = np.load('../data/TE/classify/train_data.npy')
test_data = np.load('../data/TE/classify/test_data.npy')
train_label = np.load('../data/TE/classify/train_label.npy').squeeze(1)
test_label = np.load('../data/TE/classify/test_label.npy').squeeze(1)

train_data = torch.from_numpy(train_data).float().to(device)
train_label = torch.from_numpy(train_label).float().type(torch.LongTensor).to(device)
test_data = torch.from_numpy(test_data).float().to(device)
test_label = torch.from_numpy(test_label).float().type(torch.LongTensor).to(device)


train_dataset = TensorDataset(train_data, train_label)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# dgc
def gradient_compression(tensor):
    array = tensor.numpy().flatten()
    # threshold = np.sort(np.absolute(array))[int(array.shape[0]*value)]
    threshold = 0.01
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
    model = Net(33, 22).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    accarray = np.zeros(300)
    for i in range(300):
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
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.cpu().data.numpy().squeeze()
        
        target_y = test_label.cpu().data.numpy()
        accuracy = sum(pred_y == target_y)/(len(test_label))
        
        print('epoch = {}, acc = {}'.format(i, accuracy))
        accarray[i] = accuracy

    # evaluation
    model.eval()
    pre = model(test_data)
    prediction = torch.max(F.softmax(pre), 1)[1]
    pred_y = prediction.cpu().data.numpy().squeeze()
    target_y = test_label.cpu().data.numpy()
    print('accuracy={:.4f}'.format(sum(pred_y == target_y)/4400))
    print('precision={:.4f}'.format(precision_score(target_y, pred_y, average='macro')))
    print('recall={:.4f}'.format(recall_score(target_y, pred_y, average='macro')))
    print('f1={:.4f}'.format(f1_score(target_y, pred_y, average='macro')))
    # torch.save(model.state_dict(), 'TE_classify_model.pt')
    np.save('results/dlg/dgc0.01_test_acc.npy', accarray)


if __name__ == "__main__":
    train()

















