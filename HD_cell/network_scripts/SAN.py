import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# SAN (Aime & Damas)
# ==================
















###########################################################################


# Combine SARNN (Aime & Damas)
# =======================================

#define SARNN
class SAN_RNN(nn.Module):
	def __init__(self, max_dim, sc_layers, n_filters, sequence_length, n_simp_list, degree, Laplacians, in_size, nn_layers, nn_width, output_size, dropout, conv_activation, rnn_activation):
		super().__init__()

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.max_dim = max_dim
		self.seq_length = sequence_length
		# self.simp_convlist = [nn.ModuleList() for _ in range(self.max_dim + 1)]
		self.simp_convlist = nn.ModuleList(nn.ModuleList() for _ in range(self.max_dim + 1))
		# self.batch_norm_list = [nn.ModuleList() for _ in range(self.max_dim + 1)]

		self.Laplacians = Laplacians
		self.sc_layers = sc_layers

		if sc_layers==1:
			for i in range(self.max_dim + 1):
				self.simp_convlist[i].append(simplicial_conv_indie(n_filters, degree, conv_activation, self.Laplacians[i], self.device))
		else:
			for i in range(self.max_dim + 1):
				self.simp_convlist[i].append(simplicial_conv_in(n_filters, degree, conv_activation, self.Laplacians[i], self.device))
				for _ in range(sc_layers - 2):
					self.simp_convlist[i].append(simplicial_conv(n_filters, degree, conv_activation, self.Laplacians[i], self.device))
					# self.batch_norm_list[i].append(nn.BatchNorm2d(1))

				self.simp_convlist[i].append(simplicial_conv_out(n_filters, degree, conv_activation, self.Laplacians[i], self.device))

		if nn_layers==1:
			self.RNN = nn.RNN(input_size=in_size, hidden_size=nn_width, num_layers=nn_layers, nonlinearity=rnn_activation, batch_first=True)
		else:
			self.RNN = nn.RNN(input_size=in_size, hidden_size=nn_width, num_layers=nn_layers, nonlinearity=rnn_activation, batch_first=True, dropout=dropout)

		self.linear_out = nn.Linear(nn_width, output_size)


		self.to(self.device)

	def forward(self, xs):
		big_out_list = list()
		for k in range(self.seq_length):
			output_list = list()
			for i in range(self.max_dim + 1):
				x = xs[i][:,k,:]
				for k, layer in enumerate(self.simp_convlist[i]):
					x_temp = layer(x)
					# x_temp = self.batch_norm_list[i][k](x_temp)
					x = x_temp

				output_list.append(x)
			big_out_list.append(torch.cat(output_list, 1))
		concat_output = torch.stack(big_out_list, 1)
		out, _ = self.RNN(concat_output)
		out = self.linear_out(out)[:,-1,:]

		return out


#define RNN
class RNN(nn.Module):
	def __init__(self, device, in_size, out_size, nn_width, nn_layers, rnn_activation, dropout):
		super().__init__()

		if nn_layers==1:
			self.RNN = nn.RNN(input_size=in_size, hidden_size=nn_width, num_layers=nn_layers, nonlinearity=rnn_activation, batch_first=True)
		else:
			self.RNN = nn.RNN(input_size=in_size, hidden_size=nn_width, num_layers=nn_layers, nonlinearity=rnn_activation, batch_first=True, dropout=dropout)


		self.linear_out = nn.Linear(nn_width, out_size)
		self.device = device

		self.to(self.device)

	def forward(self, x):
		out, _ = self.RNN(x)
		out = self.linear_out(out)[:,-1,:]

		return out
