import torch
import torchvision
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt


class CustomDataset(Dataset):
	def __init__(self, data, labels):
		self.data = data
		self.labels = labels

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		item = self.data[idx]
		label = self.labels[idx]
		return item, label


def main():
	torch.manual_seed(696)
	np.random.seed(696)
	train = torchvision.datasets.MNIST(
		root="./",
		train=True,
		download=True,
		transform=torchvision.transforms.Compose([
			torchvision.transforms.ToTensor()
		]))
	test = torchvision.datasets.MNIST(
		root="./",
		train=False,
		download=True,
		transform=torchvision.transforms.Compose([
			torchvision.transforms.ToTensor()
		]))

	# sns.heatmap(train.data[20000])
	# plt.show()
	# sns.heatmap(test.data[2000])
	# plt.show()

	# train_shift_val = np.random.uniform(-5, 5, len(train))
	# train_shift = torch.zeros(len(train), 28, 28)
	# for i in range(len(train)):
	# 	train_shift[i] = torch.full((28, 28), train_shift_val[i])
	# train.data = train.data.add(train_shift)
	# 
	# test_shift_val = np.random.normal(-5, 5, len(test))
	# test_shift = torch.zeros(len(test), 28, 28)
	# for i in range(len(test)):
	# 	test_shift[i] = torch.full((28, 28), test_shift_val[i])
	# test.data = test.data.add(test_shift)

	pad = np.random.randint(0, 168, (len(train), 2))
	train_pad = np.zeros((len(train), 4), dtype=int)
	train_pad[:, 0] = pad[:, 0]
	train_pad[:, 1] = 168 - pad[:, 0]
	train_pad[:, 2] = pad[:, 1]
	train_pad[:, 3] = 168 - pad[:, 1]
	train_new = torch.zeros((len(train), 196, 196))
	for sample in range(len(train)):
		curr = train_pad[sample]
		pad = (curr[0], curr[1], curr[2], curr[3])
		train_new[sample] = F.pad(train.data[sample], pad, "constant", 0)

	pad = np.random.randint(0, 168, (len(test), 2))
	test_pad = np.zeros((len(test), 4), dtype=int)
	test_pad[:, 0] = pad[:, 0]
	test_pad[:, 1] = 168 - pad[:, 0]
	test_pad[:, 2] = pad[:, 1]
	test_pad[:, 3] = 168 - pad[:, 1]
	test_new = torch.zeros((len(test), 196, 196))
	for sample in range(len(test)):
		curr = train_pad[sample]
		pad = (curr[0], curr[1], curr[2], curr[3])
		test_new[sample] = F.pad(test.data[sample], pad, "constant", 0)

	# sns.heatmap(train.data[20000])
	# plt.show()
	# sns.heatmap(train_new[20000])
	# plt.show()
	# 
	# sns.heatmap(test.data[2000])
	# plt.show()
	# sns.heatmap(test_new[2000])
	# plt.show()

	custom_train = CustomDataset(train_new, train.targets)
	custom_test = CustomDataset(test_new, test.targets)

	# Random split with fixed seed
	train_set_size = int(len(custom_train) * 0.8)
	valid_set_size = len(custom_train) - train_set_size
	custom_train, custom_validation = data.random_split(custom_train, [train_set_size, valid_set_size])
	# sns.heatmap(custom_train[12299][0])
	# plt.title(str(custom_train[12299][1]))
	# plt.show()
	# sns.heatmap(custom_test[2599][0])
	# plt.title(str(custom_test[2599][1]))
	# plt.show()

	print('Train data set:', len(custom_train))
	print('Test data set:', len(custom_test))
	print('Valid data set:', len(custom_validation))

	batch_size = 256
	train_loader = DataLoader(dataset=custom_train, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(dataset=custom_test, batch_size=batch_size, shuffle=False)
	validation_loader = DataLoader(dataset=custom_validation, batch_size=batch_size, shuffle=False)

	return train_loader, validation_loader, test_loader


if __name__ == '__main__':
	main()
