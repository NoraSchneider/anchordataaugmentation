# code base: https://github.com/huaxiuyao/C-Mixup (Yao et. al., 2022) 
# extended by forward_ada method which allows for applying ADA in the manifold. 
import torch
import torch.nn as nn
from copy import deepcopy
from copy import deepcopy
from ada import ADA
import baselines.Localmixup.localmixup as LM

class Learner(nn.Module):
	
	def __init__(self, args, hid_dim = 128, weights = None, sigmoid = None):
		super(Learner, self).__init__()
		if sigmoid: 
			self.block_1 = nn.Sequential(nn.Linear(args.input_dim, hid_dim), nn.Sigmoid())
			self.block_2 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.Sigmoid())
			self.fclayer = nn.Sequential(nn.Linear(hid_dim, 1))
		else:
			self.block_1 = nn.Sequential(nn.Linear(args.input_dim, hid_dim), nn.LeakyReLU(0.1))
			self.block_2 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.LeakyReLU(0.1))
			self.fclayer = nn.Sequential(nn.Linear(hid_dim, 1))

		
		if weights != None:
			self.load_state_dict(deepcopy(weights))

	def reset_weights(self, weights):
		self.load_state_dict(deepcopy(weights))

	def forward_mixup(self, x1, x2, lam=None):
		x1 = self.block_1(x1)
		x2 = self.block_1(x2)
		x = lam * x1 + (1 - lam) * x2
		x = self.block_2(x)
		output = self.fclayer(x)
		return output
	
	def forward_localmixup(self, x, y, eps):
		x = self.block_1(x)
		with torch.no_grad():
			x, y_a, y_b, lam, dist = LM.mixup_data(x, y, eps=eps)
		x = self.block_2(x)
		output = self.fclayer(x)
		return output, y_a, y_b, lam, dist

	def forward(self, x):
		x = self.block_1(x)
		x = self.block_2(x)
		output = self.fclayer(x)
		return output

	def repr_forward(self, x):
		with torch.no_grad():
			x = self.block_1(x)
			repr = self.block_2(x)
			return repr
		
	def forward_anchor(self, x1, gamma, anchorMatrix):
		x1 = self.block_1(x1)
		x1_til, _ = ADA.transform_pytorch(X=x1, gamma=gamma, anchorMatrix=anchorMatrix)
		x = self.block_2(x1_til)
		output = self.fclayer(x)
		return output