from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 20, 5, 1)
		self.conv2 = nn.Conv2d(20, 50, 5, 1)
		self.fc1 = nn.Linear(4*4*50, 500)
		self.fc2 = nn.Linear(501, 1)

	def forward(self, x, r):
		r = r.float()
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 4*4*50)
		x = F.relu(self.fc1(x))
		#print(x.size()) #this is where the proble is
		x = torch.cat((x,r[:x.size()[0]]),1)
		#add the element r to the vector x that has length 500 currently (torch.cat)
		x = self.fc2(x)
		return x #F.log_softmax(x, dim=1)


#model = LogisticRegressionModel()
 #that relates sigmoid
#input = (torch.randn(0,10) == input_dim)
def train(args, model, device, train_loader, optimizer, epoch):
	model.train()
	criterion = nn.BCEWithLogitsLoss()
	r = torch.randint(low=0, high=10, size=(64,1))
	r =r.float()

	#criterion = torch.nn.CrossEntropy()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
	#	print(target.size())

		target = target.float()

		#print(bool_label)
# compare r with orginal target to define bool_label
		optimizer.zero_grad()
		output = model(data,r)

		bool_label = torch.randint(low=0, high=10, size= (target.size()[0],1))#target.size())
		for i in range(target.size()[0]):
			if(r[i]==target[i]):
				bool_label[i] = 1.0
			else:
				bool_label[i] = 0.0
		bool_label = bool_label.float()

		loss = criterion(output,bool_label) #CrossEntropy(output,bool_label)#nn.BCEWithLogitsLoss(output, bool_label) #label is true or false output
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]g\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
# compare r with orginal target to define bool_label

	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			#r = torch.randint(low=0, high=1, size=(target.size()[0],1))
			r = torch.randint(low=0, high=1, size=(target.size()[0],1))
			r1 = torch.randint(low=1, high=2, size=(target.size()[0],1))
			r2 = torch.randint(low=2, high=3, size=(target.size()[0],1))
			r3 = torch.randint(low=3, high=4, size=(target.size()[0],1))
			r4 = torch.randint(low=4, high=5, size=(target.size()[0],1))
			r5 = torch.randint(low=5, high=6, size=(target.size()[0],1))
			r6 = torch.randint(low=6, high=7, size=(target.size()[0],1))
			r7 = torch.randint(low=7, high=8, size=(target.size()[0],1))
			r8 = torch.randint(low=8, high=9, size=(target.size()[0],1))
			r9 = torch.randint(low=9, high=10, size=(target.size()[0],1))

			output = torch.cat((model(data,r),model(data,r1),model(data,r2),model(data,r3),model(data,r4),model(data,r5),model(data,r6),model(data,r7),model(data,r8),model(data,r9)),1)


			#target = target.float()
			test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
			#print(output.size())
			pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
			#print(pred)

			correct += pred.eq(target.view_as(pred)).sum().item()


		#	output = torch.argmax(output, 1).reshape(target.size(),1)

			"""test_loss += criterion(output, target).item() # sum up batch loss
			pred = output.argmax(dim=10, keepdim=True) # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()"""

	test_loss /= len(test_loader.dataset)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))

def main():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
						help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=10, metavar='N',
						help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
						help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
						help='SGD momentum (default: 0.5)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
						help='how many batches to wait before logging training status')

	parser.add_argument('--save-model', action='store_true', default=False,
						help='For Saving the current Model')
	args = parser.parse_args()
	use_cuda = not args.no_cuda and torch.cuda.is_available()

	torch.manual_seed(args.seed)

	device = torch.device("cuda" if use_cuda else "cpu")

	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
	#filter o
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train=True, download=True,
					   transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])),
		batch_size=args.batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train=False, transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])),
		batch_size=args.test_batch_size, shuffle=True, **kwargs)


	model = Net().to(device)
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

	for epoch in range(1, args.epochs + 1):
		train(args, model, device, train_loader, optimizer, epoch)
		test(args, model, device, test_loader)

	if (args.save_model):
		torch.save(model.state_dict(),"mnist_cnn.pt")

if __name__ == '__main__':
	main()
