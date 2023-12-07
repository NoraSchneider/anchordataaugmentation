# code base: https://github.com/huaxiuyao/C-Mixup (Yao et. al., 2022) 
# extended by forward_ada method which allows for applying ADA in the manifold. 
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import random 
from copy import deepcopy
from torch.autograd import Variable
from ada import ADA
import baselines.Localmixup.localmixup as LM


# ---> https://github.com/mims-harvard/TDC/tree/master/
class Learner_Dti_dg(nn.Module):
	def __init__(self, hparams = None, weights = None):
		super(Learner_Dti_dg, self).__init__()

		self.num_classes = 1
		self.input_shape = [(63, 100), (26, 1000)]
		self.num_domains = 6
		self.hparams = hparams

		self.featurizer = DTI_Encoder()
		self.classifier = Classifier(
			self.featurizer.n_outputs,
			self.num_classes,
			False)
			#self.hparams['nonlinear_classifier'])

		#self.network = mySequential(self.featurizer, self.classifier)

		if weights != None:
			self.load_state_dict(deepcopy(weights))

	def reset_weights(self, weights):
		self.load_state_dict(deepcopy(weights))

	def forward(self,x):
		drug_num = self.input_shape[0][0] * self.input_shape[0][1]
		x_drug = x[:,:drug_num].reshape(-1,self.input_shape[0][0],self.input_shape[0][1])
		x_protein = x[:,drug_num:].reshape(-1,self.input_shape[1][0],self.input_shape[1][1])
		
		feature_out = self.featurizer.forward(x_drug,x_protein)
		linear_out = self.classifier(feature_out)
		return linear_out
	
	def repr_forward(self, x):
		with torch.no_grad():
			drug_num = self.input_shape[0][0] * self.input_shape[0][1]
			x_drug = x[:,:drug_num].reshape(-1,self.input_shape[0][0],self.input_shape[0][1])
			x_protein = x[:,drug_num:].reshape(-1,self.input_shape[1][0],self.input_shape[1][1])
			
			repr = self.featurizer.forward(x_drug,x_protein)
			return repr
	

	def forward_mixup(self,x1, x2, lambd):
		drug_num = self.input_shape[0][0] * self.input_shape[0][1]
		x1_drug = x1[:,:drug_num].reshape(-1,self.input_shape[0][0],self.input_shape[0][1])
		x1_protein = x1[:,drug_num:].reshape(-1,self.input_shape[1][0],self.input_shape[1][1])
		
		x2_drug = x2[:,:drug_num].reshape(-1,self.input_shape[0][0],self.input_shape[0][1])
		x2_protein = x2[:,drug_num:].reshape(-1,self.input_shape[1][0],self.input_shape[1][1])

		feature_out = self.featurizer.forward_mixup(x1_drug,x1_protein,x2_drug,x2_protein,lambd)
		linear_out = self.classifier(feature_out)
		return linear_out
		#return self.network.forward_mixup(x1_drug, x1_protein,x2_drug, x2_protein,lambd)

	def forward_localmixup(self, x, y, eps):
		drug_num = self.input_shape[0][0] * self.input_shape[0][1]
		x1_drug = x[:,:drug_num].reshape(-1,self.input_shape[0][0],self.input_shape[0][1])
		x1_protein = x[:,drug_num:].reshape(-1,self.input_shape[1][0],self.input_shape[1][1])

		feature_out = self.featurizer.forward_localmixup(x1_drug, x1_protein, y, eps)
		linear_out = self.classifier(feature_out)
		x = self.block_1(x)
		with torch.no_grad():
			x, y_a, y_b, lam, dist = LM.mixup_data(x, y, eps=eps)
		x = self.block_2(x)
		output = self.fclayer(x)
		return output, y_a, y_b, lam, dist

	def forward_anchor(self, x1, gamma, anchorMatrix):
		drug_num = self.input_shape[0][0] * self.input_shape[0][1]
		x1_drug = x1[:,:drug_num].reshape(-1,self.input_shape[0][0],self.input_shape[0][1])
		x1_protein = x1[:,drug_num:].reshape(-1,self.input_shape[1][0],self.input_shape[1][1])

		feature_out = self.featurizer.forward_anchor(x1_drug,x1_protein,gamma=gamma, anchorMatrix=anchorMatrix)
		linear_out = self.classifier(feature_out)
		return linear_out


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class CNN(nn.Sequential):
	def __init__(self, encoding):
		super(CNN, self).__init__()
		if encoding == 'drug':
			in_ch = [63] + [32,64,96]
			kernels = [4,6,8]
			self.layer_size = 3
			self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
													out_channels = in_ch[i+1], 
													kernel_size = kernels[i]) for i in range(self.layer_size)])
			self.conv = self.conv.double()
			n_size_d = self._get_conv_output((63, 100))
			self.fc1 = nn.Linear(n_size_d, 256)
		elif encoding == 'protein':
			in_ch = [26] + [32,64,96]
			kernels = [4,8,12]
			self.layer_size = 3
			self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
												out_channels = in_ch[i+1], 
												kernel_size = kernels[i]) for i in range(self.layer_size)])
			self.conv = self.conv.double()
			n_size_p = self._get_conv_output((26, 1000))
			self.fc1 = nn.Linear(n_size_p, 256)

	def _get_conv_output(self, shape):
		bs = 1
		input = Variable(torch.rand(bs, *shape))
		output_feat = self._forward_features(input.double())
		n_size = output_feat.data.view(bs, -1).size(1)
		return n_size

	def _forward_features(self, x):
		for l in self.conv:
			x = F.relu(l(x))
		x = F.adaptive_max_pool1d(x, output_size=1)
		return x
	
	def _forward_features_mixup(self, x1, x2, lambd):
		mixup_layer = random.sample(range(self.layer_size),1)[0]
		layer_ith, mix_flag = 0,0
		for l in self.conv:
			if layer_ith <= mixup_layer:
				x1 = F.relu(l(x1))
				x2 = F.relu(l(x2))
			else:
				if mix_flag == 0:
					x = x1 * lambd + x2 * (1 - lambd)
					mix_flag = 1
				x = F.relu(l(x))
			layer_ith += 1

		if mix_flag == 0:
			x = x1 * lambd + x2 * (1 - lambd)

		x = F.adaptive_max_pool1d(x, output_size=1)
		return x

	def _forward_features_anchor(self, x1, gamma, anchorMatrix):
		mixup_layer = random.sample(range(self.layer_size),1)[0]
		layer_ith, mix_flag = 0,0
		#print("start call")
		for l in self.conv:
			#print("layer ith", layer_ith)
			if layer_ith <= mixup_layer:
					#print("l")
					x1 = F.relu(l(x1))
			else:
				if mix_flag == 0:
					#print(x1.shape)
					x_til, _ = ADA.transform_pytorch(X=x1, gamma=gamma, anchorMatrix=anchorMatrix)
					x_til = x_til.double()
					#print(x_til.shape)
					mix_flag = 1
				x_til = F.relu(l(x_til))
			layer_ith += 1
		if mix_flag == 0:
				x_til, _ = ADA.transform_pytorch(X=x1, gamma=gamma, anchorMatrix=anchorMatrix)
		x_til = F.adaptive_max_pool1d(x_til, output_size=1)

		return x_til
	
	def _forward_features_localmixup(self, x1, y, eps):
		return ""


	def forward(self, v):
		v = self._forward_features(v.double())
		v = v.view(v.size(0), -1)
		v = self.fc1(v.float())
		return v
	
	def forward_mixup(self, v1, v2, lambd):
		v = self._forward_features_mixup(v1.double(),v2.double(),lambd)
		v = v.view(v.size(0), -1)
		v = self.fc1(v.float())
		return v
	
	def forward_anchor(self, v1, gamma, anchorMatrix):
		#print("forward anchor 1")
		#print(v1.dtype)
		v = self._forward_features_anchor(v1.double(), gamma, anchorMatrix)
		v = v.view(v.size(0), -1)
		v = self.fc1(v.float())
		return v

	def forward_localmixup(self, v1, y, eps):
		return ""

   
class DTI_Encoder(nn.Sequential):
	def __init__(self):
		super(DTI_Encoder, self).__init__()
		self.input_dim_drug = 256
		self.input_dim_protein = 256

		self.model_drug = CNN('drug')
		self.model_protein = CNN('protein')

		self.dropout = nn.Dropout(0.1)

		self.hidden_dims = [256, 128]
		layer_size = len(self.hidden_dims) + 1
		dims = [self.input_dim_drug + self.input_dim_protein] + self.hidden_dims + [128]
		
		self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])
		self.n_outputs = 128

	def forward(self, v_D, v_P):
		# each encoding
		v_D = self.model_drug(v_D)
		v_P = self.model_protein(v_P)
		# concatenate and classify
		v_f = torch.cat((v_D, v_P), 1)
		for i, l in enumerate(self.predictor):
			v_f = l(v_f)
		return v_f

	def forward_mixup(self, v_D1, v_P1, v_D2, v_P2, lambd):
		# each encoding
		v_D = self.model_drug.forward_mixup(v_D1,v_D2,lambd)
		v_P = self.model_protein.forward_mixup(v_P1,v_P2,lambd)
		# concatenate and classify
		v_f = torch.cat((v_D, v_P), 1)
		for i, l in enumerate(self.predictor):
			v_f = l(v_f)
		return v_f
	
	def forward_anchor(self, v_D1, v_P1, gamma, anchorMatrix):
		# each encoding
		v_D = self.model_drug.forward_anchor(v_D1,gamma, anchorMatrix)
		v_P = self.model_protein.forward_anchor(v_P1,gamma, anchorMatrix)
		# concatenate and classify
		v_f = torch.cat((v_D, v_P), 1)
		for i, l in enumerate(self.predictor):
			v_f = l(v_f)
		return v_f

	def forward_localmixup(self, v_D1, v_P1, y, eps):
		return ""


def Classifier(in_features, out_features, is_nonlinear=False):
	if is_nonlinear:
		return torch.nn.Sequential(
			torch.nn.Linear(in_features, in_features // 2),
			torch.nn.ReLU(),
			torch.nn.Linear(in_features // 2, in_features // 4),
			torch.nn.ReLU(),
			torch.nn.Linear(in_features // 4, out_features))
	else:
		return torch.nn.Linear(in_features, out_features)
