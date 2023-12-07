# code base: https://github.com/huaxiuyao/C-Mixup (Yao et. al., 2022) 
# extended by forward_ada method which allows for applying ADA in the manifold. import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from ada import ADA
from torchvision import models
import torch
import baselines.Localmixup.localmixup as LM

class Learner_RCF_MNIST(nn.Module):
	def __init__(self, args, weights = None):
		super(Learner_RCF_MNIST, self).__init__()
		self.args = args

		# get feature extractor from original model
		ori_model = models.resnet18(pretrained=True)
		#for param in model.parameters():
		#    param.requires_grad = False
		# Parameters of newly constructed modules have requires_grad=True by default
		num_ftrs = ori_model.fc.in_features
		# print(f'num_ftrs = {num_ftrs}')
		
		self.feature_extractor = torch.nn.Sequential(*list(ori_model.children())[:-2])
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # GAP
		self.fc = nn.Linear(num_ftrs, 1)

		if weights != None:
			self.load_state_dict(deepcopy(weights))

	def reset_weights(self, weights):
		self.load_state_dict(deepcopy(weights))

	def forward(self, x):
		x = self.feature_extractor(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		output = self.fc(x)
		return output

	def forward_mixup(self, x1, x2, lam = None):
		x1 = self.feature_extractor(x1)
		x2 = self.feature_extractor(x2)

		# mixup feature
		x = lam * x1 + (1 - lam) * x2
		
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		output = self.fc(x)
		return output

	def forward_localmixup(self, x, y, eps):
		x = self.feature_extractor(x)
		with torch.no_grad():
			x, y_a, y_b, lam, dist = LM.mixup_data(x, y, eps=eps)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		output = self.fc(x)
		return output, y_a, y_b, lam, dist
	
	def forward_anchor(self, x1, gamma, anchorMatrix):
		x1 = self.feature_extractor(x1)
		x1_til, _ = ADA.transform_pytorch(X=x1, gamma=gamma, anchorMatrix=anchorMatrix)
		
		x = self.avgpool(x1_til)
		x = x.view(x.size(0), -1)
		output = self.fc(x)
		return output
