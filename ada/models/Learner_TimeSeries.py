# code base: https://github.com/huaxiuyao/C-Mixup (Yao et. al., 2022) 
# extended by forward_ada method which allows for applying ADA in the manifold. 
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from ada import ADA
import baselines.Localmixup.localmixup as LM

# ---> :https://github.com/laiguokun/LSTNet
class Learner_TimeSeries(nn.Module):
	def __init__(self, args, data, weights = None):
		super(Learner_TimeSeries, self).__init__()
		self.use_cuda = args.cuda
		self.P = int(args.window)
		self.m = int(data.m)
		self.hidR = int(args.hidRNN)
		self.hidC = int(args.hidCNN)
		self.hidS = int(args.hidSkip)
		self.Ck = args.CNN_kernel
		self.skip = args.skip
		self.pt = int((self.P - self.Ck)/self.skip)
		print(f'self.pt = {self.pt}')
		self.hw = args.highway_window
		self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m))
		self.GRU1 = nn.GRU(self.hidC, self.hidR)
		self.dropout = nn.Dropout(p = args.dropout)
		if (self.skip > 0):
			self.GRUskip = nn.GRU(self.hidC, self.hidS)
			print(self.hidR + self.skip * self.hidS, self.m)
			self.linear1 = nn.Linear(int(self.hidR + self.skip * self.hidS), self.m)
		else:
			self.linear1 = nn.Linear(self.hidR, self.m)
		if (self.hw > 0): #highway -> autoregressiion
			self.highway = nn.Linear(self.hw, 1)
		self.output = None
		if (args.output_fun == 'sigmoid'):
			self.output = F.sigmoid
		if (args.output_fun == 'tanh'):
			self.output = F.tanh

		if weights != None:
			self.load_state_dict(deepcopy(weights))

	def reset_weights(self, weights):
		self.load_state_dict(deepcopy(weights))

	def forward(self, x):
		batch_size = x.size(0)
		#CNN
		c = x.view(-1, 1, self.P, self.m)
		c = F.relu(self.conv1(c))
		c = self.dropout(c)
		c = torch.squeeze(c, 3)
		
		# RNN time number <-> layer number
		r = c.permute(2, 0, 1).contiguous()
		_, r = self.GRU1(r)
		r = self.dropout(torch.squeeze(r,0))

		
		#skip-rnn
		
		if (self.skip > 0):
			s = c[:,:, int(-self.pt * self.skip):].contiguous()
			s = s.view(batch_size, self.hidC, int(self.pt), int(self.skip))
			s = s.permute(2,0,3,1).contiguous()
			s = s.view(int(self.pt), int(batch_size * self.skip), int(self.hidC))
			_, s = self.GRUskip(s)
			s = s.view(batch_size, int(self.skip * self.hidS))
			s = self.dropout(s)
			r = torch.cat((r,s),1)
		
		# FC
		res = self.linear1(r)
		
		#highway auto-regression
		if (self.hw > 0):
			z = x[:, -self.hw:, :]
			z = z.permute(0,2,1).contiguous().view(-1, self.hw)
			z = self.highway(z)
			z = z.view(-1,self.m)
			res = res + z
			
		if (self.output):
			res = self.output(res)

		return res

	def repr_forward(self, x):
		batch_size = x.size(0)
		#CNN
		c = x.view(-1, 1, self.P, self.m)
		c = F.relu(self.conv1(c))
		c = self.dropout(c)
		c = torch.squeeze(c, 3)
		
		# RNN time number <-> layer number
		r = c.permute(2, 0, 1).contiguous()
		_, r = self.GRU1(r)
		r = self.dropout(torch.squeeze(r,0))

		
		#skip-rnn
		
		if (self.skip > 0):
			s = c[:,:, int(-self.pt * self.skip):].contiguous()
			s = s.view(batch_size, self.hidC, int(self.pt), int(self.skip))
			s = s.permute(2,0,3,1).contiguous()
			s = s.view(int(self.pt), int(batch_size * self.skip), int(self.hidC))
			_, s = self.GRUskip(s)
			s = s.view(batch_size, int(self.skip * self.hidS))
			s = self.dropout(s)
			r = torch.cat((r,s),1)
		
		# FC
		return r
		res = self.linear1(r)
		
		#highway auto-regression
			
	def forward_localmixup(self, x, y, eps):
		batch_size = x.size(0)
		#CNN
		c1 = x.view(-1, 1, self.P, self.m)
		c1 = F.relu(self.conv1(c1))
		c1 = self.dropout(c1)
		c1 = torch.squeeze(c1, 3)
		
		with torch.no_grad():
			c, y_a, y_b, lam, dist = LM.mixup_data(c1, y, eps=eps)
		
		# RNN time number <-> layer number
		r = c.permute(2, 0, 1).contiguous()
		_, r = self.GRU1(r)
		r = self.dropout(torch.squeeze(r,0))

		#skip-rnn
		
		if (self.skip > 0):
			s = c[:,:, int(-self.pt * self.skip):].contiguous()
			s = s.view(batch_size, self.hidC, int(self.pt), int(self.skip))
			s = s.permute(2,0,3,1).contiguous()
			s = s.view(int(self.pt), int(batch_size * self.skip), int(self.hidC))
			_, s = self.GRUskip(s)
			s = s.view(batch_size, int(self.skip * self.hidS))
			s = self.dropout(s)
			r = torch.cat((r,s),1)
		
		# FC
		res = self.linear1(r)
		
		#highway auto-regression --> not mixup
		if (self.hw > 0):
			x, _, _, _, _ = LM.mixup_data(x, y, eps=eps)
			z = x[:, -self.hw:, :]
			z = z.permute(0,2,1).contiguous().view(-1, self.hw)
			z = self.highway(z)
			z = z.view(-1,self.m)
			res = res + z
			
		if (self.output):
			res = self.output(res)
		return res, y_a, y_b, lam, dist


	def forward_mixup(self, x1, x2, lam):
		batch_size = x1.size(0)
		#CNN
		c1 = x1.view(-1, 1, self.P, self.m)
		c1 = F.relu(self.conv1(c1))
		c1 = self.dropout(c1)
		c1 = torch.squeeze(c1, 3)

		#CNN
		c2 = x2.view(-1, 1, self.P, self.m)
		c2 = F.relu(self.conv1(c2))
		c2 = self.dropout(c2)
		c2 = torch.squeeze(c2, 3)
		
		# just mixup after conv block
		c = lam * c1 + (1 - lam) * c2

		# RNN time number <-> layer number
		r = c.permute(2, 0, 1).contiguous()
		_, r = self.GRU1(r)
		r = self.dropout(torch.squeeze(r,0))

		#skip-rnn
		
		if (self.skip > 0):
			s = c[:,:, int(-self.pt * self.skip):].contiguous()
			s = s.view(batch_size, self.hidC, int(self.pt), int(self.skip))
			s = s.permute(2,0,3,1).contiguous()
			s = s.view(int(self.pt), int(batch_size * self.skip), int(self.hidC))
			_, s = self.GRUskip(s)
			s = s.view(batch_size, int(self.skip * self.hidS))
			s = self.dropout(s)
			r = torch.cat((r,s),1)
		
		# FC
		res = self.linear1(r)
		
		#highway auto-regression --> not mixup
		if (self.hw > 0):
			x = lam * x1 + (1 - lam) * x2
			z = x[:, -self.hw:, :]
			z = z.permute(0,2,1).contiguous().view(-1, self.hw)
			z = self.highway(z)
			z = z.view(-1,self.m)
			res = res + z
			
		if (self.output):
			res = self.output(res)
		return res

	def forward_anchor(self, x1, gamma, anchorMatrix):
		
		batch_size = x1.size(0)
		#CNN
		c1 = x1.view(-1, 1, self.P, self.m)
		c1 = F.relu(self.conv1(c1))
		c1 = self.dropout(c1)
		c1 = torch.squeeze(c1, 3)

		c_til, _ = ADA.transform_pytorch(X=c1, gamma=gamma, anchorMatrix=anchorMatrix)

		# RNN time number <-> layer number
		r = c_til.permute(2, 0, 1).contiguous()
		_, r = self.GRU1(r)
		r = self.dropout(torch.squeeze(r,0))

		#skip-rnn
		
		if (self.skip > 0):
			s = c_til[:,:, int(-self.pt * self.skip):].contiguous()
			s = s.view(batch_size, self.hidC, int(self.pt), int(self.skip))
			s = s.permute(2,0,3,1).contiguous()
			s = s.view(int(self.pt), int(batch_size * self.skip), int(self.hidC))
			_, s = self.GRUskip(s)
			s = s.view(batch_size, int(self.skip * self.hidS))
			s = self.dropout(s)
			r = torch.cat((r,s),1)
		
		# FC
		res = self.linear1(r)
		
		#highway auto-regression --> not mixup
		if (self.hw > 0):
			x, _ = ADA.transform_pytorch(X=x1, gamma=gamma, anchorMatrix=anchorMatrix)
			z = x[:, -self.hw:, :]
			z = z.permute(0,2,1).contiguous().view(-1, self.hw)
			z = self.highway(z)
			z = z.view(-1,self.m)
			res = res + z
			
		if (self.output):
			res = self.output(res)
		return res