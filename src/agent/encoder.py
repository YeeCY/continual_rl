import torch
import torch.nn as nn
from utils import tie_weights


OUT_DIM = {2: 39, 4: 35, 6: 31, 8: 27, 10: 23, 11: 21, 12: 19}


class CenterCrop(nn.Module):
	"""Center-crop if observation is not already cropped"""
	def __init__(self, size):
		super().__init__()
		assert size == 84
		self.size = size

	def forward(self, x):
		assert x.ndim == 4, 'input must be a 4D tensor'
		if x.size(2) == self.size and x.size(3) == self.size:
			return x
		elif x.size(-1) == 100:
			return x[:, :, 8:-8, 8:-8]
		else:
			return ValueError('unexepcted input size')


class NormalizeImg(nn.Module):
	"""Normalize observation"""
	def forward(self, x):
		return x / 255.


class PixelEncoder(nn.Module):
	"""Convolutional encoder of pixel observations"""
	def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32):
		super().__init__()
		assert len(obs_shape) == 3

		self.feature_dim = feature_dim
		self.num_layers = num_layers
		# TODO (chongyi zheng): delete this line
		# self.num_shared_layers = num_shared_layers

		# (chongyi zheng): add shape indicators in the comment
		self.preprocess = nn.Sequential(
			CenterCrop(size=84), NormalizeImg()
		)  # if x.shape = (N, 9, 100, 100), convert it to (N, 9, 84, 84). Then normalize pixel value to [0, 1].

		self.convs = nn.ModuleList(
			[nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
		)  # (N, 9, 84, 84) -> (N, 32, 41, 41)
		for i in range(num_layers - 1):  # (N, 32, 41, 41) -> (N, 32, 39, 39) -> (N, 32, 37, 37) ...
			self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

		out_dim = OUT_DIM[num_layers]
		self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)  # (N, 32 * out_dim * out_dim) -> (N, 100)
		self.ln = nn.LayerNorm(self.feature_dim)

		self.head = nn.Sequential(
			nn.Linear(num_filters * out_dim * out_dim, self.feature_dim),
			nn.LayerNorm(self.feature_dim))

		self.outputs = dict()  # log placeholder

	# TODO (chongyi zheng): delete this version of 'forward_conv'
	# def forward_conv(self, obs, detach=False):
	# 	obs = self.preprocess(obs)
	# 	conv = torch.relu(self.convs[0](obs))
	# 	for i in range(1, self.num_layers):
	# 		conv = torch.relu(self.convs[i](conv))
	# 		if i == self.num_shared_layers - 1 and detach:
	# 			conv = conv.detach()
	#
	# 	h = conv.view(conv.size(0), -1)
	# 	return h

	def forward_conv(self, obs):
		obs = self.preprocess(obs)
		self.outputs['obs'] = obs

		conv = torch.relu(self.convs[0](obs))
		self.outputs['conv1'] = conv
		for i in range(1, self.num_layers):
			conv = torch.relu(self.convs[i](conv))
			self.outputs['conv%s' % (i)] = conv

		h = conv.view(conv.size(0), -1)
		return h

	def forward(self, obs, detach=False):
		h = self.forward_conv(obs)

		if detach:
			h = h.detach()  # stop gradient propagation to convolutional layers

		out = self.head(h)
		out = torch.tanh(out)
		self.outputs['out'] = out

		return out

	# TODO (chongyi zheng): delete this version of 'copy_conv_weights_from'
	# def copy_conv_weights_from(self, source, n=None):
	# 	"""Tie n first convolutional layers"""
	# 	if n is None:
	# 		n = self.num_layers
	# 	for i in range(n):
	# 		tie_weights(src=source.convs[i], trg=self.convs[i])
	def copy_conv_weights_from(self, source):
		for i in range(self.num_layers):
			tie_weights(src=source.convs[i], trg=self.convs[i])

	def log(self, logger, step):
		for k, v in self.outputs.items():
			logger.log_histogram(f'train_encoder/{k}_hist', v, step)
			if len(v.shape) > 2:
				logger.log_image(f'train_encoder/{k}_img', v[0], step)

		for i in range(self.num_layers):
			logger.log_param(f'train_encoder/conv{i}', self.convs[i], step)


# TODO (chongyi zheng): delete function 'make_encoder'
# def make_encoder(obs_shape, feature_dim, num_layers, num_filters, num_shared_layers):
# 	assert num_layers in OUT_DIM.keys(), 'invalid number of layers'
# 	if num_shared_layers == -1 or num_shared_layers == None:
# 		num_shared_layers = num_layers
# 	assert num_shared_layers <= num_layers and num_shared_layers > 0, \
# 		f'invalid number of shared layers, received {num_shared_layers} layers'
# 	return PixelEncoder(
# 		obs_shape, feature_dim, num_layers, num_filters, num_shared_layers
# 	)
