# -*- coding: utf-8 -*-
'''
MINIST: 28*28 -> 100 -> ReLu ->10 Linear 
'''
import pdb
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")

## NN Model
class RubinNN(nn.Module):
	"""docstring for RubinNN"nn.Module"""
	def __init__(self):  #  define all layers
		super(RubinNN,self).__init__()
			#归一。。view()
		self.network = nn.Sequential(
			nn.Flatten(),
			nn.Linear(28*28, 100),
			nn.ReLU(),
			nn.Linear(100,10),
			# nn.ReLU(),
			# nn.Linear(512,10),
			# nn.Softmax(),
			# nn.pool(),
			# nn.Conv2d(),
			# nn.Dropout()						
			)

	def forward(self, xb): 
		logits = self.network(xb)
		return logits


## Training function

def train(model, loss_fn, opt, train_dl):
	model.train()
	for batch, (xb, yb) in enumerate(train_dl):
		xb, yb = xb.to(dev), yb.to(dev)		
		y = model(xb)
		loss = loss_fn(y, yb)
		loss.backward()
		opt.step()
		opt.zero_grad()
		pdb.set_trace()
		if batch % 100 == 0:
			loss = loss.item()
			print(f"loss:{loss:>7f}")
		
def valid(model, loss_fn, valid_dl):
	batch_num = len(valid_dl)
	size = len(valid_dl.dataset)
	total_loss, correct = 0, 0

	model.eval()
	with torch.no_grad():
		for xb, yb in valid_dl:
			xb, yb = xb.to(dev), yb.to(dev)
			y = model(xb)
			loss = loss_fn(y, yb)
			total_loss += loss
			correct += (y.argmax(1) == yb).type(torch.float).sum().item()
	test_loss = total_loss/batch_num
	correct /= size
	print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# training dataset prepration
train_ds = datasets.MNIST(
	root = 'data',
	train = True,
	download = True,
	transform = ToTensor(),
	#target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
	)
valid_ds = datasets.MNIST(
	root = 'data',
	train = False,
	download = True,
	transform = ToTensor(),
	#target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
	)

# HyperParameters settings
lr = 1e-3
epochs = 10
bs = 64
	            ## Begin of Training
train_dl = DataLoader(train_ds, batch_size = bs, shuffle = True)
valid_dl = DataLoader(valid_ds, batch_size = bs*2)

model = RubinNN().to(dev)



#loss_fn = F.cross_entropy
loss_fn = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr)

for epoch in range(epochs):
	print(f"Epoch {epoch + 1}\n-------------------------------")
	train(model, loss_fn, opt, train_dl)
	valid(model, loss_fn, valid_dl)
				## End of Training


# ## Save Model
# torch.save(model,'RubinNN.pth')
# ## Load Model 
# model = torch.load('RubinNN.pth')




