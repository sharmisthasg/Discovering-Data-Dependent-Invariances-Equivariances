# import torch
# import numpy as np
# import itertools
# import pandas as pd
# from torch.nn import functional as F
# import torch.nn as nn
# import seaborn as sns
# from matplotlib import pyplot as plt
# from scipy.optimize import linear_sum_assignment
#
#
# class Weight(nn.Module):
# 	def __init__(self):
# 		super(Weight, self).__init__()
# 		torch.manual_seed(696)
# 		self.w = torch.nn.Parameter(torch.randn(2))
#
# 	def forward(self, x, A):
# 		W = torch.matmul(A, self.w)
# 		W = W.reshape((10, 10))
# 		op = torch.matmul(x, W)
# 		op = F.relu(op)
# 		op = torch.mean(op, dim=-1)
# 		return op
#
#
# class A(nn.Module):
# 	def __init__(self):
# 		super(A, self).__init__()
# 		torch.manual_seed(696)
# 		a = torch.randn(100, 2)
# 		self.A = torch.nn.Parameter(torch.exp(nn.functional.log_softmax(a, 1)))
#
# 	def forward(self, x, w):
# 		W = torch.matmul(self.A, w)
# 		W = W.reshape((10, 10))
# 		op = torch.matmul(x, W)
# 		op = F.relu(op)
# 		op = torch.mean(op, dim=-1)
# 		return op
#
#
# def train(model, optimizer, X, y, device, A):
# 	criterion = torch.nn.L1Loss(reduction='mean')
# 	model.train()
# 	total_loss = 0
# 	optimizer.zero_grad()
# 	prediction = model(X, A)
# 	loss = criterion(prediction, y) + 1e-2 * torch.sum(torch.square(model.w))
# 	loss.backward()
# 	optimizer.step()
# 	total_loss += loss.item()
#
# 	return total_loss
#
#
# def validate(model, optimizer, X, y, device, w):
# 	model.train()
# 	criterion = torch.nn.L1Loss(reduction='mean')
# 	total_loss = 0
# 	optimizer.zero_grad()
# 	prediction = model(X, w)
# 	AA = torch.exp(nn.functional.log_softmax(model.A, 1))
# 	loss_reg = torch.trace(torch.sqrt(AA.T.mm(AA))) / AA.shape[0]
# 	loss_reg += -2 * \
# 	            torch.sum(torch.log(AA + 1e-6) * (AA + 1e-6), -1).mean() * 0.5
# 	loss = criterion(prediction, y) + loss_reg
# 	loss.backward()
# 	optimizer.step()
# 	total_loss += loss.item()
# 	# print(model.A)
#
# 	return total_loss
#
#
# def test(A, w, X, y, device):
# 	criterion = torch.nn.L1Loss(reduction='mean')
# 	A_indices = A.argmax(-1)
# 	A_val = torch.zeros((100, 2), dtype=torch.float64)
# 	A_val[torch.arange(100), A_indices] = 1
# 	W = torch.matmul(A_val, w)
# 	W = W.reshape((10, 10))
# 	op = torch.matmul(X, W)
# 	prediction = F.relu(op)
# 	prediction = torch.mean(prediction, dim=-1)
# 	return criterion(prediction, y)
#
#
# def adj_to_set(A):
# 	num_obj, num_cluster = A.shape
# 	cluster_all = np.argmax(A, 1)
# 	all_sets = [set() for _ in range(num_cluster)]
# 	for obj_idx, cluster_idx in enumerate(cluster_all):
# 		all_sets[cluster_idx].add(obj_idx)
# 	return all_sets
#
#
# def partition_distance(A1, A2):
# 	S1 = adj_to_set(A1)
# 	S2 = adj_to_set(A2)
# 	cost = np.zeros((len(S1), len(S2)))
# 	for j in range(len(S1)):
# 		for k in range(len(S2)):
# 			cost[j, k] = -1 * len(S1[j] & S2[k])
# 	row_ind, col_ind = linear_sum_assignment(cost)
# 	return len(A1) + cost[row_ind, col_ind].sum()
#
#
# def main():
# 	torch.manual_seed(696)
# 	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 	D = 10  # input dimension
# 	C = 1
# 	# X = torch.rand(N, D).to(device)  # (N, D)
# 	X = torch.tensor(pd.read_csv("../Dataset/Training.csv", usecols=list(range(10))).to_numpy()).double().to(device)
# 	# y = torch.sum(X, axis=-1).reshape(-1, C)  # (N, C)
# 	y = torch.tensor(pd.read_csv("../Dataset/Training.csv", usecols=[10]).to_numpy()).reshape(-1, C).double().to(device)
# 	X_val = torch.tensor(pd.read_csv("../Dataset/Validation.csv", usecols=list(range(10))).to_numpy()).double().to(
# 		device)
# 	y_val = torch.tensor(pd.read_csv("../Dataset/Validation.csv", usecols=[10]).to_numpy()).reshape(-1, C).double().to(
# 		device)
#
# 	lr_train = 1e-2  # Learning rate
# 	lr_val = 1e-1
#
# 	t_model = Weight().to(device)
# 	t_model.to(device)
# 	t_model = t_model.double()
# 	t_optimizer = torch.optim.Adam(t_model.parameters(), lr=lr_train)  # optimizer
#
# 	v_model = A().to(device)
# 	v_model.to(device)
# 	v_model = v_model.double()
# 	v_optimizer = torch.optim.Adam(v_model.parameters(), lr=lr_val)  # optimizer
#
# 	tt_model = Weight().to(device)
# 	tt_model.to(device)
# 	tt_model = tt_model.double()
#
# 	vv_model = A().to(device)
# 	vv_model.to(device)
# 	vv_model = vv_model.double()
#
# 	torch.save(t_model.state_dict(), "../models/t_model" + ".pt")
# 	torch.save(v_model.state_dict(), "../models/v_model" + ".pt")
#
# 	optimum_training_loss = float('inf')
# 	optimum_validation_loss = float('inf')
# 	for epoch in range(50):
# 		vv_model.load_state_dict(torch.load("../models/v_model" + ".pt"))
# 		for inner in range(75):
# 			training_loss = train(t_model, t_optimizer, X, y, device, vv_model.A)
# 			print(f'inner {inner} training loss: {training_loss}')
# 			if training_loss < optimum_training_loss:
# 				optimum_training_loss = training_loss
# 				torch.save(t_model.state_dict(), "../models/t_model" + ".pt")
# 		tt_model.load_state_dict(torch.load("../models/t_model" + ".pt"))
# 		for outer in range(50):
# 			validation_loss = validate(v_model, v_optimizer, X_val, y_val, device, tt_model.w)
# 			print(f'outer {outer} validation loss:{validation_loss}')
# 			if validation_loss < optimum_validation_loss:
# 				optimum_validation_loss = validation_loss
# 				torch.save(v_model.state_dict(), "../models/v_model" + ".pt")
#
# 	X_test = torch.tensor(pd.read_csv("../Dataset/Test.csv", usecols=list(range(10))).to_numpy()).double().to(device)
# 	y_test = torch.tensor(pd.read_csv("../Dataset/Test.csv", usecols=[10]).to_numpy()).reshape(-1, C).double().to(
# 		device)
# 	t_model.load_state_dict(torch.load("../models/t_model" + ".pt"))
# 	v_model.load_state_dict(torch.load("../models/v_model" + ".pt"))
#
# 	test_loss = test(v_model.A, t_model.w, X_test, y_test, device)
#
# 	# print(model.encoder.linear.weight)
# 	# sns.heatmap(model.encoder.linear.weight.detach().numpy())
# 	# plt.show()
# 	#
# 	# print(model.encoder.linear1.weight)
# 	# sns.heatmap(model.encoder.linear1.weight.detach().numpy())
# 	# plt.show()
# 	#
# 	# print(model.decoder.linear.weight)
# 	# sns.heatmap(model.decoder.linear.weight.detach().numpy())
# 	# plt.show()
#
# 	# x = torch.diagonal(torch.matmul(model.weight, model.eq_matrix), offset=-1, dim1=1, dim2=2)
# 	# print(torch.sum(torch.sum(torch.abs(x), dim=-1), dim=-1))
# 	# sns.heatmap(model.eq_matrix.reshape((100, 100)).detach().numpy())
# 	# plt.show()
#
# 	A_indices = v_model.A.argmax(-1)
# 	A_val = torch.zeros((100, 2), dtype=torch.float64)
# 	A_val[torch.arange(100), A_indices] = 1
# 	print(A_val)
# 	sns.heatmap(A_val.detach().numpy())
# 	plt.show()
# 	print(t_model.w)
# 	W = torch.matmul(A_val, t_model.w)
# 	print(W)
# 	sns.heatmap(W.detach().numpy().reshape(10, 10))
# 	plt.show()
# 	print(f'test loss:{test_loss}')
#
# 	true_A = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
# 	                   [0, 1], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
# 	                   [0, 1], [0, 1], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
# 	                   [0, 1], [0, 1], [0, 1], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
# 	                   [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
# 	                   [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1],
# 	                   [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [0, 1], [0, 1], [0, 1],
# 	                   [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [0, 1], [0, 1],
# 	                   [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [0, 1],
# 	                   [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0]])
#
# 	print('Parition distance:%s' % partition_distance(A_val, true_A))
#
#
# if __name__ == '__main__':
# 	main()


# import torch
# import numpy as np
# import itertools
# import pandas as pd
# from torch.nn import functional as F
# import torch.nn as nn
# import seaborn as sns
# from matplotlib import pyplot as plt
# from scipy.optimize import linear_sum_assignment
#
#
# class Weight(nn.Module):
# 	def __init__(self):
# 		super(Weight, self).__init__()
# 		torch.manual_seed(696)
# 		self.w = torch.nn.Parameter(torch.randn(2))
# 		self.w1 = torch.nn.Parameter(torch.randn(10,2))
#
# 	def forward(self,x,A):
# 		op = F.relu(torch.matmul(x,self.w1))
# 		W = torch.matmul(A,self.w).reshape((2,2))
# 		op = F.relu(torch.matmul(op,W))
# 		op = torch.mean(op,dim=-1)
# 		return op
#
#
# class A(nn.Module):
# 	def __init__(self):
# 		super(A, self).__init__()
# 		torch.manual_seed(696)
# 		a = torch.randn(4, 2)
# 		self.A = torch.nn.Parameter(torch.exp(nn.functional.log_softmax(a, 1)))
#
# 	def forward(self,x,w,w1):
# 		op = F.relu(torch.matmul(x,w1))
# 		W = torch.matmul(self.A,w).reshape((2,2))
# 		op = F.relu(torch.matmul(op,W))
# 		op = torch.mean(op,dim=-1)
# 		return op
#
#
# def train(model, optimizer, X, y, device, A):
# 	criterion = torch.nn.L1Loss(reduction='mean')
# 	model.train()
# 	total_loss = 0
# 	optimizer.zero_grad()
# 	prediction = model(X, A)
# 	loss = criterion(prediction, y) + 1e-2 * torch.sum(torch.abs(model.w)) + 1e-2 * torch.sum(torch.square(model.w1))
# 	loss.backward()
# 	optimizer.step()
# 	total_loss += loss.item()
#
# 	return total_loss
#
#
# def validate(model, optimizer, X, y, device, w, w1):
# 	model.train()
# 	criterion = torch.nn.L1Loss(reduction='mean')
# 	total_loss = 0
# 	optimizer.zero_grad()
# 	prediction = model(X, w, w1)
# 	AA = torch.exp(nn.functional.log_softmax(model.A, 1))
# 	loss_reg = 0
# 	loss_reg += torch.trace(torch.sqrt(AA.T.mm(AA))) / AA.shape[0]
# 	loss_reg += -1 * \
# 	            torch.sum(torch.log(AA + 1e-6) * (AA + 1e-6), -1).mean() * 0.5
# 	loss = criterion(prediction, y) + loss_reg
# 	loss.backward()
# 	optimizer.step()
# 	total_loss += loss.item()
#
# 	return total_loss
#
#
# def test(A, w, w1, X, y, device):
# 	criterion = torch.nn.L1Loss(reduction='mean')
# 	# A_indices = A.argmax(-1)
# 	# A_val = torch.zeros((4, 2), dtype=torch.float64)
# 	# A_val[torch.arange(4), A_indices] = 1
# 	W = torch.matmul(A, w).reshape((2,2))
#
# 	op = F.relu(torch.matmul(X,w1))
# 	op = F.relu(torch.matmul(op,W))
# 	op = torch.mean(op,dim=-1)
# 	return criterion(op, y)
#
#
# def main():
# 	torch.manual_seed(696)
# 	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 	D = 10  # input dimension
# 	C = 1
# 	X = torch.tensor(pd.read_csv("../Dataset/Training.csv", usecols=list(range(10))).to_numpy()).double().to(device)
# 	y = torch.tensor(pd.read_csv("../Dataset/Training.csv", usecols=[10]).to_numpy()).reshape(-1, C).double().to(device)
# 	X_val = torch.tensor(pd.read_csv("../Dataset/Validation.csv", usecols=list(range(10))).to_numpy()).double().to(
# 		device)
# 	y_val = torch.tensor(pd.read_csv("../Dataset/Validation.csv", usecols=[10]).to_numpy()).reshape(-1, C).double().to(
# 		device)
#
# 	lr_train = 1e-2  # Learning rate
# 	lr_val = 1e-1
#
# 	t_model = Weight().to(device)
# 	t_model.to(device)
# 	t_model = t_model.double()
# 	t_optimizer = torch.optim.Adam(t_model.parameters(), lr=lr_train)  # optimizer
#
# 	v_model = A().to(device)
# 	v_model.to(device)
# 	v_model = v_model.double()
# 	v_optimizer = torch.optim.Adam(v_model.parameters(), lr=lr_val)  # optimizer
#
# 	tt_model = Weight().to(device)
# 	tt_model.to(device)
# 	tt_model = tt_model.double()
#
# 	vv_model = A().to(device)
# 	vv_model.to(device)
# 	vv_model = vv_model.double()
#
# 	torch.save(t_model.state_dict(), "../models/t_model" + ".pt")
# 	torch.save(v_model.state_dict(), "../models/v_model" + ".pt")
#
# 	optimum_training_loss = float('inf')
# 	optimum_validation_loss = float('inf')
# 	for epoch in range(100):
# 		vv_model.load_state_dict(torch.load("../models/v_model" + ".pt"))
# 		for inner in range(75):
# 			training_loss = train(t_model, t_optimizer, X, y, device, vv_model.A)
# 			print(f'inner {inner} training loss: {training_loss}')
# 			if training_loss < optimum_training_loss:
# 				optimum_training_loss = training_loss
# 				torch.save(t_model.state_dict(), "../models/t_model" + ".pt")
#
# 		tt_model.load_state_dict(torch.load("../models/t_model" + ".pt"))
# 		for outer in range(50):
# 			validation_loss = validate(v_model, v_optimizer, X_val, y_val, device, tt_model.w, tt_model.w1)
# 			print(f'outer {outer} validation loss:{validation_loss}')
# 			if validation_loss < optimum_validation_loss:
# 				optimum_validation_loss = validation_loss
# 				torch.save(v_model.state_dict(), "../models/v_model" + ".pt")
#
# 	X_test = torch.tensor(pd.read_csv("../Dataset/Test.csv", usecols=list(range(10))).to_numpy()).double().to(device)
# 	y_test = torch.tensor(pd.read_csv("../Dataset/Test.csv", usecols=[10]).to_numpy()).reshape(-1, C).double().to(
# 		device)
# 	t_model.load_state_dict(torch.load("../models/t_model" + ".pt"))
# 	v_model.load_state_dict(torch.load("../models/v_model" + ".pt"))
#
# 	test_loss = test(v_model.A, t_model.w, tt_model.w1, X_test, y_test, device)
#
# 	print(t_model.w1)
# 	print(v_model.A)
# 	A_indices = v_model.A.argmax(-1)
# 	A_val = torch.zeros((4, 2), dtype=torch.float64)
# 	A_val[torch.arange(4), A_indices] = 1
# 	print(A_val)
# 	sns.heatmap(A_val.detach().numpy())
# 	plt.show()
# 	print(t_model.w)
# 	W = torch.matmul(A_val, t_model.w)
# 	print(W.reshape(2,2))
# 	sns.heatmap(W.detach().numpy().reshape(2, 2))
# 	plt.show()
# 	print(f'test loss:{test_loss}')
#
#
# if __name__ == '__main__':
# 	main()


# import torch
# import numpy as np
# import itertools
# import pandas as pd
# from torch.nn import functional as F
# import torch.nn as nn
# import seaborn as sns
# from matplotlib import pyplot as plt
# from scipy.optimize import linear_sum_assignment
#
#
# class Weight(nn.Module):
# 	def __init__(self):
# 		super(Weight, self).__init__()
# 		torch.manual_seed(696)
# 		self.w = torch.nn.Parameter(torch.randn(2))
# 		# self.w1 = torch.nn.Parameter(torch.randn(10,2))
#
# 	def forward(self,x,A):
# 		A = torch.exp(nn.functional.softmax(A, dim=-1))
# 		W = torch.matmul(A,self.w).reshape((10,10))
# 		op = F.relu(torch.matmul(x,W))
# 		op = torch.sum(op,dim=-1)
# 		return op
#
#
# class A(nn.Module):
# 	def __init__(self):
# 		super(A, self).__init__()
# 		torch.manual_seed(696)
# 		a = torch.randn(100, 2)
# 		self.A = torch.nn.Parameter(torch.exp(nn.functional.softmax(a, dim=-1)))
#
# 	def forward(self,x,w):
# 		A = torch.exp(nn.functional.softmax(self.A, dim=-1))
# 		W = torch.matmul(A,w).reshape((10,10))
# 		op = F.relu(torch.matmul(x,W))
# 		op = torch.sum(op,dim=-1)
# 		return op
#
#
# def train(model, optimizer, X, y, device, A):
# 	criterion = torch.nn.L1Loss(reduction='mean')
# 	model.train()
# 	total_loss = 0
# 	optimizer.zero_grad()
# 	prediction = model(X, A)
# 	loss = criterion(prediction, y) + 1e-1 * torch.sum(torch.abs(model.w))
# 	loss.backward()
# 	optimizer.step()
# 	total_loss += loss.item()
#
# 	return total_loss
#
#
# def validate(model, optimizer, X, y, device, w):
# 	model.train()
# 	criterion = torch.nn.L1Loss(reduction='mean')
# 	total_loss = 0
# 	optimizer.zero_grad()
# 	prediction = model(X, w)
# 	# AA = torch.exp(nn.functional.softmax(model.A, dim=-1))
# 	AA = model.A
# 	loss_reg = 0
# 	# loss_reg += torch.trace(torch.sqrt(AA.T.mm(AA))) / AA.shape[0]
# 	# loss_reg += torch.trace(AA) / AA.shape[0]
# 	# loss_reg += torch.sum(torch.sqrt(AA.T.mm(AA)) / AA.shape[0])
# 	# loss_reg += -1 * \
# 	#             torch.sum(torch.exp(AA + 1e-6) * (AA + 1e-6), -1).mean() * 0.5
# 	loss = criterion(prediction, y) + 0.05 * loss_reg
# 	loss.backward()
# 	optimizer.step()
# 	total_loss += loss.item()
#
# 	return total_loss
#
#
# def test(A, w, X, y, device):
# 	criterion = torch.nn.L1Loss(reduction='mean')
# 	# A_indices = A.argmax(-1)
# 	# A_val = torch.zeros((4, 2), dtype=torch.float64)
# 	# A_val[torch.arange(4), A_indices] = 1
# 	A = torch.exp(nn.functional.softmax(A, dim=-1))
# 	W = torch.matmul(A, w).reshape((10,10))
#
# 	op = F.relu(torch.matmul(X,W))
# 	op = torch.sum(op,dim=-1)
# 	return criterion(op, y)
#
#
# def main():
# 	torch.manual_seed(696)
# 	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 	D = 10  # input dimension
# 	C = 1
# 	X = torch.tensor(pd.read_csv("../Dataset/Training.csv", usecols=list(range(10))).to_numpy()).double().to(device)
# 	y = torch.tensor(pd.read_csv("../Dataset/Training.csv", usecols=[10]).to_numpy()).reshape(-1, C).double().to(device)
# 	X_val = torch.tensor(pd.read_csv("../Dataset/Validation.csv", usecols=list(range(10))).to_numpy()).double().to(
# 		device)
# 	y_val = torch.tensor(pd.read_csv("../Dataset/Validation.csv", usecols=[10]).to_numpy()).reshape(-1, C).double().to(
# 		device)
# 	X_vall = torch.tensor(pd.read_csv("../Dataset/Val.csv", usecols=list(range(10))).to_numpy()).double().to(
# 		device)
# 	y_vall = torch.tensor(pd.read_csv("../Dataset/Val.csv", usecols=[10]).to_numpy()).reshape(-1, C).double().to(
# 		device)
#
# 	lr_train = 1e-4  # Learning rate
# 	lr_val = 1e-3
#
# 	t_model = Weight().to(device)
# 	t_model.to(device)
# 	t_model = t_model.double()
# 	t_optimizer = torch.optim.Adam(t_model.parameters(), lr=lr_train)  # optimizer
#
# 	v_model = A().to(device)
# 	v_model.to(device)
# 	v_model = v_model.double()
# 	v_optimizer = torch.optim.Adam(v_model.parameters(), lr=lr_val)  # optimizer
#
# 	tt_model = Weight().to(device)
# 	tt_model.to(device)
# 	tt_model = tt_model.double()
#
# 	vv_model = A().to(device)
# 	vv_model.to(device)
# 	vv_model = vv_model.double()
#
# 	torch.save(t_model.state_dict(), "../models/t_model" + ".pt")
# 	torch.save(v_model.state_dict(), "../models/v_model" + ".pt")
#
# 	optimum_val_loss = float('inf')
# 	optimum_validation_loss = float('inf')
# 	for epoch in range(10):
# 		vv_model.load_state_dict(torch.load("../models/v_model" + ".pt"))
# 		for inner in range(50):
# 			training_loss = train(t_model, t_optimizer, X, y, device, vv_model.A)
# 			val_loss = test(vv_model.A, t_model.w, X_vall, y_vall, device)
# 			print(f'inner {inner} training loss: {training_loss} val_loss: {val_loss}')
# 			if val_loss < optimum_val_loss:
# 				optimum_val_loss = val_loss
# 				torch.save(t_model.state_dict(), "../models/t_model" + ".pt")
#
# 		tt_model.load_state_dict(torch.load("../models/t_model" + ".pt"))
# 		for outer in range(75):
# 			validation_loss = validate(v_model, v_optimizer, X_val, y_val, device, tt_model.w)
# 			val_loss = test(v_model.A, tt_model.w, X_vall, y_vall, device)
# 			print(f'outer {outer} validation loss:{validation_loss} val_loss: {val_loss}')
# 			if val_loss < optimum_val_loss:
# 				optimum_val_loss = val_loss
# 				torch.save(v_model.state_dict(), "../models/v_model" + ".pt")
# 	#
# 	X_test = torch.tensor(pd.read_csv("../Dataset/Test.csv", usecols=list(range(10))).to_numpy()).double().to(device)
# 	y_test = torch.tensor(pd.read_csv("../Dataset/Test.csv", usecols=[10]).to_numpy()).reshape(-1, C).double().to(
# 		device)
# 	t_model.load_state_dict(torch.load("../models/t_model" + ".pt"))
# 	v_model.load_state_dict(torch.load("../models/v_model" + ".pt"))
#
# 	test_loss = test(v_model.A, t_model.w, X_test, y_test, device)
# 	W = torch.matmul(v_model.A, t_model.w)
# 	sns.heatmap(W.detach().numpy().reshape((10,10)))
# 	plt.show()
#
# 	print(W)
#
# 	print(v_model.A)
# 	sns.heatmap(v_model.A.detach().numpy())
# 	plt.show()
#
# 	A_indices = v_model.A.argmax(-1)
# 	A_val = torch.zeros((100, 2), dtype=torch.float64)
# 	A_val[torch.arange(100), A_indices] = 1
# 	print(A_val)
# 	sns.heatmap(A_val.detach().numpy())
# 	plt.show()
# 	print(t_model.w)
# 	W = torch.matmul(A_val, t_model.w)
# 	print(W.reshape(10,10))
# 	sns.heatmap(W.detach().numpy().reshape(10, 10))
# 	plt.show()
# 	print(f'test loss:{test_loss}')
#
#
# if __name__ == '__main__':
# 	main()


import torch
import numpy as np
import itertools
import pandas as pd
from torch.nn import functional as F
import torch.nn as nn
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment


class Weight(nn.Module):
	def __init__(self):
		super(Weight, self).__init__()
		torch.manual_seed(696)
		self.w = nn.Parameter(torch.randn(1,1))

	def forward(self,x,A):
		A = nn.Parameter(torch.exp(nn.functional.softmax(A, 1)))

		x_expanded = x.reshape((x.shape[0] * x.shape[1], 1))
		# print('x_ex',x_expanded.shape)
		W = torch.matmul(self.w,A.T)
		# W = W.reshape((10,10))
		features = F.relu(torch.matmul(x_expanded,W))
		# print('f1',features.shape)
		features = torch.sum(features,dim=-1)
		# print('f2',features.shape)
		features = features.reshape(x.shape[0],10)
		# print('f3',features.shape)
		features = torch.sum(features,dim=-1)
		# print('final',features.shape)
		return features


class A(nn.Module):
	def __init__(self):
		super(A, self).__init__()
		torch.manual_seed(696)
		a = torch.randn(25,1)
		self.A = nn.Parameter(torch.exp(nn.functional.softmax(a, 1)))

	def forward(self,x,w):
		A = nn.Parameter(torch.exp(nn.functional.softmax(self.A, 1)))

		x_expanded = x.reshape((x.shape[0] * x.shape[1], 1))
		W = torch.matmul(w, A.T)
		# W = W.reshape((10, 10))
		features = F.relu(torch.matmul(x_expanded, W))
		features = torch.sum(features, dim=-1)
		features = features.reshape(x.shape[0], 10)
		features = torch.sum(features, dim=-1)
		return features


def train(model, optimizer, X, y, device, A):
	criterion = torch.nn.L1Loss(reduction='mean')
	model.train()
	total_loss = 0
	optimizer.zero_grad()
	prediction = model(X, A)
	loss = criterion(prediction, y) + 1e-1 * torch.sum(torch.square(model.w))
	loss.backward()
	optimizer.step()
	total_loss += loss.item()

	return total_loss


def validate(model, optimizer, X, y, device, w):
	model.train()
	criterion = torch.nn.L1Loss(reduction='mean')
	total_loss = 0
	optimizer.zero_grad()
	prediction = model(X, w)
	# AA = torch.exp(nn.functional.softmax(model.A, dim=-1))
	AA = model.A
	loss_reg = 0
	loss_reg += torch.trace(torch.sqrt(AA.T.mm(AA))) / AA.shape[0]
	# loss_reg += torch.trace(AA) / AA.shape[0]
	# loss_reg += torch.sum(torch.sqrt(AA.T.mm(AA)) / AA.shape[0])
	# loss_reg += -1 * \
	#             torch.sum(torch.exp(AA + 1e-6) * (AA + 1e-6), -1).mean() * 0.5
	loss = criterion(prediction, y) + 0.05 * loss_reg
	loss.backward()
	optimizer.step()
	total_loss += loss.item()

	return total_loss


def test(A, w, X, y, device):
	criterion = torch.nn.L1Loss(reduction='mean')
	# A_indices = A.argmax(-1)
	# A_val = torch.zeros((4, 2), dtype=torch.float64)
	# A_val[torch.arange(4), A_indices] = 1
	A = torch.exp(nn.functional.softmax(A, dim=1))
	x_expanded = X.reshape((X.shape[0] * X.shape[1], 1))
	W = torch.matmul(w, A.T)
	# W = W.reshape((10, 10))
	features = F.relu(torch.matmul(x_expanded, W))
	features = torch.sum(features, dim=-1)
	features = features.reshape(X.shape[0], 10)
	features = torch.sum(features, dim=-1)
	return criterion(features, y), features


def main():
	torch.manual_seed(696)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	D = 10  # input dimension
	C = 1
	X = torch.tensor(pd.read_csv("../Dataset/Training.csv", usecols=list(range(10))).to_numpy()).double().to(device)
	y = torch.tensor(pd.read_csv("../Dataset/Training.csv", usecols=[10]).to_numpy()).reshape(-1, C).double().to(device)
	X_val = torch.tensor(pd.read_csv("../Dataset/Validation.csv", usecols=list(range(10))).to_numpy()).double().to(
		device)
	y_val = torch.tensor(pd.read_csv("../Dataset/Validation.csv", usecols=[10]).to_numpy()).reshape(-1, C).double().to(
		device)
	X_vall = torch.tensor(pd.read_csv("../Dataset/Val.csv", usecols=list(range(10))).to_numpy()).double().to(
		device)
	y_vall = torch.tensor(pd.read_csv("../Dataset/Val.csv", usecols=[10]).to_numpy()).reshape(-1, C).double().to(
		device)

	lr_train = 1e-3  # Learning rate
	lr_val = 1e-2

	t_model = Weight().to(device)
	t_model.to(device)
	t_model = t_model.double()
	t_optimizer = torch.optim.Adam(t_model.parameters(), lr=lr_train)  # optimizer

	v_model = A().to(device)
	v_model.to(device)
	v_model = v_model.double()
	v_optimizer = torch.optim.Adam(v_model.parameters(), lr=lr_val)  # optimizer

	tt_model = Weight().to(device)
	tt_model.to(device)
	tt_model = tt_model.double()

	vv_model = A().to(device)
	vv_model.to(device)
	vv_model = vv_model.double()

	torch.save(t_model.state_dict(), "../models/t_model" + ".pt")
	torch.save(v_model.state_dict(), "../models/v_model" + ".pt")

	optimum_val_loss = float('inf')
	optimum_validation_loss = float('inf')
	for epoch in range(10):
		vv_model.load_state_dict(torch.load("../models/v_model" + ".pt"))
		for inner in range(20):
			training_loss = train(t_model, t_optimizer, X, y, device, vv_model.A)
			val_loss, preds = test(vv_model.A, t_model.w, X_vall, y_vall, device)
			print(f'inner {inner} training loss: {training_loss} val_loss: {val_loss}')
			if val_loss < optimum_val_loss:
				optimum_val_loss = val_loss
				torch.save(t_model.state_dict(), "../models/t_model" + ".pt")

		tt_model.load_state_dict(torch.load("../models/t_model" + ".pt"))
		for outer in range(75):
			validation_loss = validate(v_model, v_optimizer, X_val, y_val, device, tt_model.w)
			val_loss, preds = test(v_model.A, tt_model.w, X_vall, y_vall, device)
			print(f'outer {outer} validation loss:{validation_loss} val_loss: {val_loss}')
			if val_loss < optimum_val_loss:
				optimum_val_loss = val_loss
				torch.save(v_model.state_dict(), "../models/v_model" + ".pt")
	#
	X_test = torch.tensor(pd.read_csv("../Dataset/Test.csv", usecols=list(range(10))).to_numpy()).double().to(device)
	y_test = torch.tensor(pd.read_csv("../Dataset/Test.csv", usecols=[10]).to_numpy()).reshape(-1, C).double().to(
		device)
	t_model.load_state_dict(torch.load("../models/t_model" + ".pt"))
	v_model.load_state_dict(torch.load("../models/v_model" + ".pt"))

	test_loss, preds = test(v_model.A, t_model.w, X_test, y_test, device)
	W = torch.matmul(t_model.w,v_model.A.T)
	sns.heatmap(W.detach().numpy().reshape((5,5)))
	plt.show()

	print(W)

	print(v_model.A)
	sns.heatmap(v_model.A.detach().numpy())
	plt.show()

	A_indices = v_model.A.argmax(-1)
	A_val = torch.zeros((25, 2), dtype=torch.float64)
	A_val[torch.arange(25), A_indices] = 1
	print(A_val)
	sns.heatmap(A_val.detach().numpy())
	plt.show()
	print(t_model.w)
	W = torch.matmul(t_model.w, A_val.T)
	print(W.reshape(5,5))
	sns.heatmap(W.detach().numpy().reshape(5, 5))
	plt.show()
	print(f'test loss:{test_loss}')
	print(f'preds:{preds}')


if __name__ == '__main__':
	main()


