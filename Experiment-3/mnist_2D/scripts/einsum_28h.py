import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logging
import numpy as np
import warnings

warnings.filterwarnings('always')
from collections import OrderedDict
from torch.nn import functional as F
import sys

sys.path.append('../datasets')
import Load_static_augmented_dataset


class MLP(torch.nn.Module):
	def __init__(self):
		super(MLP, self).__init__()
		weight1 = torch.randn(28, 28, 56, 56)
		self.weight1 = torch.nn.Parameter(weight1)
		weight2 = torch.randn(28, 28, 56, 56)
		self.weight2 = torch.nn.Parameter(weight2)
		weight3 = torch.randn(28, 28, 56, 56)
		self.weight3 = torch.nn.Parameter(weight3)
		weight4 = torch.randn(28, 28, 56, 56)
		self.weight4 = torch.nn.Parameter(weight4)
		weight5 = torch.randn(28, 28, 56, 56)
		self.weight5 = torch.nn.Parameter(weight5)
		weight6 = torch.randn(28, 28, 56, 56)
		self.weight6 = torch.nn.Parameter(weight6)
		weight7 = torch.randn(28, 28, 56, 56)
		self.weight7 = torch.nn.Parameter(weight7)
		weight8 = torch.randn(28, 28, 56, 56)
		self.weight8 = torch.nn.Parameter(weight8)
		weight9 = torch.randn(28, 28, 56, 56)
		self.weight9 = torch.nn.Parameter(weight9)
		weight10 = torch.randn(28, 28, 56, 56)
		self.weight10 = torch.nn.Parameter(weight10)

	def forward(self, x):
		# print(f'x:{x.shape}')
		output1 = torch.einsum('ijmn,kmn->kij', self.weight1, x)
		output1 = F.elu(output1, 1)
		# output1 = torch.tanh(output1)
		output1 = output1.reshape((output1.shape[0], 1, output1.shape[1], output1.shape[2]))
		output2 = torch.einsum('ijmn,kmn->kij', self.weight2, x)
		output2 = F.elu(output2, 1)
		# output2 = torch.tanh(output2)
		output2 = output2.reshape((output2.shape[0], 1, output2.shape[1], output2.shape[2]))
		output3 = torch.einsum('ijmn,kmn->kij', self.weight3, x)
		output3 = F.elu(output3, 1)
		# output3 = torch.tanh(output3)
		output3 = output3.reshape((output3.shape[0], 1, output3.shape[1], output3.shape[2]))
		output4 = torch.einsum('ijmn,kmn->kij', self.weight4, x)
		output4 = F.elu(output4, 1)
		# output4 = torch.tanh(output4)
		output4 = output4.reshape((output4.shape[0], 1, output4.shape[1], output4.shape[2]))
		output5 = torch.einsum('ijmn,kmn->kij', self.weight5, x)
		output5 = F.elu(output5, 1)
		# output5 = torch.tanh(output5)
		output5 = output5.reshape((output5.shape[0], 1, output5.shape[1], output5.shape[2]))
		output6 = torch.einsum('ijmn,kmn->kij', self.weight6, x)
		output6 = F.elu(output6, 1)
		# output6 = torch.tanh(output6)
		output6 = output6.reshape((output6.shape[0], 1, output6.shape[1], output6.shape[2]))
		output7 = torch.einsum('ijmn,kmn->kij', self.weight7, x)
		output7 = F.elu(output7, 1)
		# output7 = torch.tanh(output7)
		output7 = output7.reshape((output7.shape[0], 1, output7.shape[1], output7.shape[2]))
		output8 = torch.einsum('ijmn,kmn->kij', self.weight8, x)
		output8 = F.elu(output8, 1)
		# output8 = torch.tanh(output8)
		output8 = output8.reshape((output8.shape[0], 1, output8.shape[1], output8.shape[2]))
		output9 = torch.einsum('ijmn,kmn->kij', self.weight9, x)
		output9 = F.elu(output9, 1)
		# output9 = torch.tanh(output9)
		output9 = output9.reshape((output9.shape[0], 1, output9.shape[1], output9.shape[2]))
		output10 = torch.einsum('ijmn,kmn->kij', self.weight10, x)
		output10 = F.elu(output10, 1)
		# output10 = torch.tanh(output10)
		output10 = output10.reshape((output10.shape[0], 1, output10.shape[1], output10.shape[2]))
		# print(f'output10 shape: {output10.shape}')
		output = torch.cat((output1, output2, output3, output4, output5, output6, output7, output8, output9, output10),
						   1)
		# print(f'output shape: {output.shape}')
		s, _ = torch.max(output, dim=-1)
		# print(f's shape: {s.shape}')
		s, _ = torch.max(s, dim=-1)
		# print(f's shape: {s.shape}')
		return s


def custom_loss_fn(model, criterion, y_pred, y, hp=1e-4):
	weights = [model.weight1, model.weight2, model.weight3, model.weight4, model.weight5, model.weight6, model.weight7,
			   model.weight8, model.weight9, model.weight10]
	total_diff = 0
	# for c in range(10):
	# 	count = 28 * 28
	# 	offset = 16
	# 	diff = 0
	# 	weight = weights[c].reshape((count, 56, 56))
	# 	curr = weight[0]
	# 	for i in range(offset - 1, count + 1, offset):
	# 		diff += torch.sum(torch.abs(weight[i] - curr))
	# 		curr = weight[i]
	#
	# 	total_diff += diff

	count = 28 * 28
	offset = 49
	subset = count // offset
	for c in range(10):
		start = np.random.randint(0, count - 1)
		diff = 0
		weight = weights[c].reshape((count,56,56))
		curr = start
		for i in range(subset):
			next = (curr + offset) % count
			diff += torch.sum(torch.abs(weight[next] - curr))
			curr = next
		total_diff += diff

	return criterion(y_pred, y) + hp * total_diff


def train(model, criterion, optimizer, train_loader, device):
	model.train()
	total_loss = 0
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		prediction = model(data)
		loss = custom_loss_fn(model, criterion, prediction, target)
		loss.backward()
		optimizer.step()
		total_loss += loss.item()

	return total_loss


def test(model, criterion, loader, device, validation):
	model.eval()
	total_loss = 0
	accuracy = []
	with torch.no_grad():
		for data, target in loader:
			data, target = data.to(device), target.to(device)
			prediction = model(data)
			if validation:
				loss = criterion(prediction, target)
				total_loss += loss
			else:
				predictions = prediction.argmax(dim=1, keepdim=True)
				accuracy.append(accuracy_score(target.cpu(), predictions.cpu()))
	if validation:
		return total_loss
	else:
		return accuracy


def main():
	torch.manual_seed(696)
	train_loader, validation_loader, test_loader = Load_static_augmented_dataset.main()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	lr = 1e-4  # Learning rate

	model = MLP().to(device)

	criterion = torch.nn.CrossEntropyLoss()  # loss function
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer

	# training the model for 1000 epochs
	optimum_training_loss = float('inf')
	optimum_validation_loss = float('inf')

	model.load_state_dict(torch.load("../models/2D_mnist_mlp_trial_28" + ".pt"))
	for epoch in range(40):
		training_loss = train(model, criterion, optimizer, train_loader, device)
		validation_loss = test(model, criterion, validation_loader, device, validation=True)
		print(f'training loss: {training_loss}, validation loss:{validation_loss}')
		if training_loss < optimum_training_loss and validation_loss < optimum_validation_loss:
			optimum_validation_loss = validation_loss
			optimum_training_loss = training_loss
			torch.save(model.state_dict(), "../models/2D_mnist_mlp_trial_28" + ".pt")

	model.load_state_dict(torch.load("../models/2D_mnist_mlp_trial_28" + ".pt"))
	accuracy = test(model, criterion, test_loader, device, validation=False)
	print(f'Test score: {sum(accuracy) / len(accuracy)}')


if __name__ == '__main__':
	main()
