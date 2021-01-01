from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pdb
#import matplotlib.pyplot as plt

#where train Loader used to be
image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size

 # load files
train_data = np.loadtxt("mnist_train.csv", delimiter=",")
test_data = np.loadtxt( "mnist_test.csv" , delimiter=",")

fac = 0.99 / 255 #convert grayscale values to values between 0.1 and 1, no 0 because they can cause weights not to update
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01
train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

# manipulate accuracy


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 20, 5, 1)
		self.conv2 = nn.Conv2d(20, 50, 5, 1)
		self.fc1 = nn.Linear(4*4*50, 500)
		self.fc2 = nn.Linear(500, 10)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 4*4*50)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

def train(model, data,target, optimizer, epoch, unchanging_guess):
	model.train()
	criterion = nn.BCEWithLogitsLoss() # binary loss shit
	#stuff to import
	data = torch.from_numpy(data)
	data = data.reshape(60000,1,28,28)
	data = data.double()
	target = torch.from_numpy(target)


	#actually net shit
	for batch_idk in range(0, data.size()[0],64):
		#find the end and begining of the batch in the dataset
		if(batch_idk+ 64 > data.size()[0]):
			end = data.size()[0]
		else:
			end = batch_idk+64
		t = target[batch_idk:end]
		r = unchanging_guess[batch_idk:end]
		d = data[batch_idk:end]
		#create the matrix that gets multiplied
		target = target.float()
		#make the true labeling for true and false

		r = r.long()
		l = torch.randint(low =0,high = 1, size = (t.size()[0],10)) #this returns if the random label is true or false
		for i in range(l.size()[0]):
			l[i][r[i][0]] = 1.0
		#training begins
		optimizer.zero_grad()
		output = model(d.float())

		#multiply
		l=l.float()
		output = output*l
		output = output.sum(1,keepdim=True)

		t=t.long()
		r = r.long()
		#boolean labels
		bool_label = torch.randint(low=0, high=10, size=(t.size()[0],1))#not efficent
		for i in range(t.size()[0]):
			if(r[i]==t[i]):
				bool_label[i] = 1.0
			else:
				bool_label[i] = 0.0

		bool_label=bool_label.float()

		#calculate loss and evalute weights
		loss = criterion(output, bool_label)#bool_label) # change
		#pdb.set_trace()
		loss.backward()
		optimizer.step()


		if batch_idk % 10 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idk, len(data),
			100. * batch_idk / len(data), loss.item()))


def test(model,data,target,large):
		model.eval()# OPTIMIZE:
		test_loss = 0
		correct = 0
		data = torch.from_numpy(data)
		data = data.reshape(10000,1,28,28)
		target = torch.from_numpy(target)
		with torch.no_grad():
			output = model(data.float())
			test_loss += F.nll_loss(output, target.flatten().long(), reduction='sum').item() # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
			target = target.long()
			#correct += pred.eq(target.view_as(pred)).sum().item()


			for i in range(target.size()[0]):
				if target[i]== pred[i]:
					correct+=1

		if(large < correct):
			large = correct
		print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			test_loss, correct, 10000,
			100. * correct / 10000))
		print(large)
		return large

def main():
	# Training settings
	Epochs = 10
	momentum = .05
	lr = .01
	unchanging_guess = torch.randint(low=0, high=10, size=(60000, 1))


	model = Net().float()
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
	large = 0
	c = 0
	#g = target.size()[0]/Epochs

	for epoch in range(1, Epochs + 1):
		train(model, train_imgs, train_labels, optimizer, epoch,unchanging_guess)
		large = test(model, test_imgs,test_labels,large)


if __name__ == '__main__':
	main()
