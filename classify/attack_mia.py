import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from scipy import spatial

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)

# set argparse
parser = argparse.ArgumentParser(description='mia for TE.')
parser.add_argument('--protection', type=int, default=1, help='whether protect or not, 0 is not.')
parser.add_argument('--multiple', type=int, default=1, help='noise multiple.')
parser.add_argument('--sigma', type=int, default=0.3, help='the std of noise.')
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
        out = torch.relu(out)
        out = self.hidden2(out)
        out = torch.relu(out)
        out = self.hidden3(out)
        out = torch.relu(out)
        out = self.predict(out)

        return out


# mian mia
def mi_face(label_index, model, num_iterations, gradient_step):
    model.to(device)
    model.eval()

    # initialize two 33 tensors with zeros

    tensor = ((torch.zeros(33)+torch.ones(33))/2).unsqueeze(0).to(device)
    image = ((torch.zeros(33)+torch.ones(33))/2).unsqueeze(0).to(device)

    # initialize with infinity
    min_loss = float("inf")

    for i in range(num_iterations):
        tensor.requires_grad = True

        # get the prediction probs
        pred = model(tensor)

        # calculate the loss and gardient for the class we want to reconstruct
        crit = nn.CrossEntropyLoss()
        loss = crit(pred, torch.tensor([label_index]).to(device))

        print('Loss: ' + str(loss.item()))

        loss.backward()

        with torch.no_grad():
            # apply gradient descent
            tensor = (tensor - gradient_step * tensor.grad)
            tensor = torch.clamp(tensor, 0, 1)

        # set image = tensor only if the new loss is the min from all iterations
        if loss < min_loss:
            min_loss = loss
            image = tensor.detach().clone().to('cpu')

    return image


def perform_attack_and_print_all_results(model, iterations):
    gradient_step_size = 0.0005
    random.seed(7)

    # average evaluation
    avg_mse = 0
    avg_cos = 0

    for j in range(22):
        print('\nReconstructing Class ' + str(j + 1))
        train_data_median = np.load('../data/TE/classify/train_data_median.npy')
        original = train_data_median[j]

        # reconstruct respective class
        reconstruction = mi_face(j, model, iterations, gradient_step_size)
        reconstruction = reconstruction.squeeze().detach().numpy().reshape(-1, 1)
        reconstruction = reconstruction.squeeze(1)

        print(original)
        print(reconstruction)

        # evaluation two vector
        rmse = np.sqrt(mean_squared_error(original, reconstruction))
        cos = 1 - spatial.distance.cosine(original, reconstruction)

        print('\nThe mean_squared_error is {}'.format(rmse))
        print('\nThe cosine_similarity is {}'.format(cos))

        avg_mse = (avg_mse + rmse)
        avg_cos = (avg_cos + cos)

        # np.save('./results/mia/reverse_result/' + str(j) + '.npy', reconstruction)
        np.save('./results/mia/reverse_result_protected/' + str(j) + '.npy', reconstruction)

    print('\nThe average mean_squared_error is {}'.format(avg_mse / 22))
    print('\nThe average cosine_similarity is {}'.format(avg_cos / 22))


model = MLP(33, 22).to(device)
parameters = torch.load('./TE_classify_model.pt')
# -------------------------加噪声----------------------------
if args.protection == 1:
    # weight1 = parameters['hidden1.weight'].cpu().numpy()
    # weight1_new = weight1 + args.multiple*np.random.normal(0, args.sigma, weight1.shape)
    # parameters['hidden1.weight'] = torch.tensor(weight1_new).float().to(device)
    #
    # bias1 = parameters['predict.bias'].cpu().numpy()
    # bias1_new = bias1 + args.multiple * np.random.normal(0, args.sigma, bias1.shape)
    # parameters['predict.bias'] = torch.tensor(bias1_new).float().to(device)

    # weight2 = parameters['hidden2.weight'].cpu().numpy()
    # weight2_new = weight2 + args.multiple*np.random.normal(0, args.sigma, weight2.shape)
    # parameters['hidden2.weight'] = torch.tensor(weight2_new).float().to(device)

    # bias2 = parameters['predict.bias'].cpu().numpy()
    # bias2_new = bias2 + args.multiple * np.random.normal(0, args.sigma, bias2.shape)
    # parameters['predict.bias'] = torch.tensor(bias2_new).float().to(device)

    # weight3 = parameters['hidden3.weight'].cpu().numpy()
    # weight3_new = weight3 + args.multiple*np.random.normal(0, args.sigma, weight3.shape)
    # parameters['hidden3.weight'] = torch.tensor(weight3_new).float().to(device)

    # bias3 = parameters['predict.bias'].cpu().numpy()
    # bias3_new = bias3 + args.multiple * np.random.normal(0, args.sigma, bias3.shape)
    # parameters['predict.bias'] = torch.tensor(bias3_new).float().to(device)

    weight4 = parameters['predict.weight'].cpu().numpy()
    weight4_new = weight4 + args.multiple * np.random.normal(0, args.sigma, weight4.shape)
    parameters['predict.weight'] = torch.tensor(weight4_new).float().to(device)

    bias4 = parameters['predict.bias'].cpu().numpy()
    bias4_new = bias4 + args.multiple * np.random.normal(0, args.sigma, bias4.shape)
    parameters['predict.bias'] = torch.tensor(bias4_new).float().to(device)

# -----------------------------------------------------
model.load_state_dict(parameters)
perform_attack_and_print_all_results(model, 30)

# ---------------测试加噪声之后的模型的准确率---------------
print('测试模型加噪声之后的分类性能')
test_data = np.load('../data/TE/classify/test_data.npy')
test_label = np.load('../data/TE/classify/test_label.npy').squeeze(1)
test_data = torch.from_numpy(test_data).float()
test_label = torch.from_numpy(test_label).float().type(torch.LongTensor)

model.eval()
model.cpu()
pre = model(test_data)
prediction = torch.max(F.softmax(pre), 1)[1]
pred_y = prediction.data.numpy().squeeze()
target_y = test_label.data.numpy()
print('accuracy={:.4f}'.format(sum(pred_y == target_y) / 4400))
print(F.softmax(model(test_data[4399])))
