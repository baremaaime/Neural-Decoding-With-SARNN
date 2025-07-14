import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# SAN (Aime & Damas)
# ==================

BATCH_SIZE = 32

class SAN(torch.nn.Module):
    def __init__(self, in_size, out_size, k=1,dropout=0.5,alpha_leaky_relu=0.2):
        self.in_size = in_size
        self.out_size = out_size
        self.k = k
        self.dropout = dropout

        super().__init__()
        self.leaky_relu = torch.nn.LeakyReLU(alpha_leaky_relu)
        self.W = torch.nn.parameter.Parameter(torch.randn(k, in_size, out_size))
        self.A = torch.nn.parameter.Parameter(torch.randn(k, in_size, out_size))
        self.R = torch.nn.parameter.Parameter(torch.randn(k, in_size, out_size))
        self.D = torch.nn.parameter.Parameter(torch.randn(k, in_size, out_size))
        self.Wh = torch.nn.parameter.Parameter(torch.randn(k, in_size, out_size))

        #attributes Z0

        self.att_l0_1 = torch.nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))
        self.att_l0_2 = torch.nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))

        #attributes Z1
        self.att_l1_1 = torch.nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))
        self.att_l1_2 = torch.nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))
        self.att_l1_3 = torch.nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))
        self.att_l1_4 = torch.nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))

        #attributes Z2
        self.att_l2_1 = torch.nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))
        self.att_l2_2 = torch.nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))

        # Initialize weights with Xavier
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.xavier_uniform_(self.A)
        torch.nn.init.xavier_uniform_(self.R)
        torch.nn.init.xavier_uniform_(self.D)
        torch.nn.init.xavier_uniform_(self.Wh)

        torch.nn.init.xavier_uniform_(self.att_l0_1)
        torch.nn.init.xavier_uniform_(self.att_l0_2)

        torch.nn.init.xavier_uniform_(self.att_l1_1)
        torch.nn.init.xavier_uniform_(self.att_l1_2)
        torch.nn.init.xavier_uniform_(self.att_l1_3)
        torch.nn.init.xavier_uniform_(self.att_l1_4)

        torch.nn.init.xavier_uniform_(self.att_l2_1)
        torch.nn.init.xavier_uniform_(self.att_l2_2)
    
    
    def compute_projection_matrix(self,L, eps, kappa):

        P = (torch.eye(L.shape[0])).to(L.device) - eps*L
        for _ in range(kappa):
            P = P @ P  # approximate the limit
        return P

    def E_f(self,X,W,K,L,attr,dropout,b=None,t=False):
            # (ExE) x (ExF_in) x (F_inxF_out) -> (ExF_out)
        
        if b!=None and t==True:
            X = b.T@X
        if b!=None and t==False:
            X = b@X
    
    
        X_f = torch.cat([X @ W[k,:,:] for k in range(K)], dim=1)

        L = L.to_dense()

        # Broadcast add
        E = self.leaky_relu((X_f @ attr[:self.out_size*K, :]) + (
             X_f @ attr[self.out_size*K:, :]).T) 
             
        zero_vec = -9e15*torch.ones_like(E)
        E = torch.where(L != 0, E, zero_vec)

        # Broadcast add
        L_f = F.dropout(F.softmax(E, dim=1), dropout, training=self.training) # (ExE) -> (ExE)

        return L_f

    def compute_z0(self, z0, z1, b1, l0_sparse_1, l0_sparse_2,l0):
        z0 = z0.float()

        for j in range(0, self.k):
            l0_sparse_1_j = torch.linalg.matrix_power(l0_sparse_1, j)
            l0_sparse_2_j = torch.linalg.matrix_power(l0_sparse_2, j)
            if j == 0:
                first_term = l0_sparse_1_j@z0@self.W[j,:,:]
                second_term = ((b1.T@l0_sparse_2_j.T).T)@z1@self.A[j,:,:]
            else:
                first_term += l0_sparse_1_j@z0@self.W[j,:,:]
                second_term += ((b1.T@l0_sparse_2_j.T).T)@z1@self.A[j,:,:]
        
        P = self.compute_projection_matrix(l0,0.1,1)
        #sparsify P
    
            
        harm = P@z0@self.Wh[0,:,:]
        return torch.sigmoid(first_term + second_term + harm)

    def compute_z1(self, z0, z1, z2, b1, b2, l1_sparse_1, l1_sparse_2, l1_sparse_3, l1_sparse_4,l1):
        for j in range(0, self.k):
            l1_sparse_1_j = torch.linalg.matrix_power(l1_sparse_1, j+1)
            l1_sparse_2_j = torch.linalg.matrix_power(l1_sparse_2, j+1)
            l1_sparse_3_j = torch.linalg.matrix_power(l1_sparse_3, j+1)
            l1_sparse_4_j = torch.linalg.matrix_power(l1_sparse_4, j+1)
            if j == 0:
                first_term = l1_sparse_1_j@z1@self.W[j,:,:]
                second_term =((b1@l1_sparse_2_j.T).T)@z0@self.A[j,:,:]
                third_term = l1_sparse_3_j@z1@self.R[j,:,:]
                fourth_term = ((b2.T@l1_sparse_4_j.T).T)@z2@self.D[j,:,:]
                    
            else:
                first_term += l1_sparse_1_j@z1@self.W[j,:,:]
                second_term +=((b1@l1_sparse_2_j.T).T)@z0@self.A[j,:,:]
                third_term += l1_sparse_3_j@z1@self.R[j,:,:]
                fourth_term += ((b2.T@l1_sparse_4_j.T).T)@z2@self.D[j,:,:]
        
        P = self.compute_projection_matrix(l1,0.1,1)
        #sparsify P
            
        harm = P@z1@self.Wh[0,:,:]
        return torch.sigmoid(first_term + second_term + third_term + harm)

    def compute_z2(self, z1, z2, b2, l2_sparse_1, l2_sparse_2,l2):

        for j in range(0, self.k):
            l2_sparse_1_j = torch.linalg.matrix_power(l2_sparse_1, j+1)
            l2_sparse_2_j = torch.linalg.matrix_power(l2_sparse_2, j+1)
            if j == 0:
                first_term = l2_sparse_1@z2@self.W[0,:,:]
                second_term = ((b2@l2_sparse_2.T).T)@z1@self.R[0,:,:]
            else:
                first_term += l2_sparse_1_j@z2@self.W[j,:,:]
                second_term += ((b2@l2_sparse_2_j.T).T)@z1@self.D[j,:,:]
        
        P = self.compute_projection_matrix(l2,0.1,1)

        harm = P@z2@self.Wh[0,:,:]
        return torch.sigmoid(first_term + second_term + harm)


    def forward(self, z0, z1, z2, b1, b2):
        """
        b1 tupla (b1_index, b1_val)
        b2 tupla (b2_index, b2_val)
        """
        n_nodes = z0.shape[0]
        n_edges = z1.shape[0]
        n_triangles = z2.shape[0]

        b1_sparse = torch.sparse_coo_tensor(b1[0], b1[1], size=(n_nodes, n_edges))
        b2_sparse = torch.sparse_coo_tensor(b2[0], b2[1], size=(n_edges, n_triangles))
        
        l0_sparse = b1_sparse@(b1_sparse.t())
        l1_d_sparse = (b1_sparse.t())@b1_sparse
        l1_u_sparse = b2_sparse@(b2_sparse.t())
        l1_sparse = l1_d_sparse + l1_u_sparse
        l2_sparse = (b2_sparse.t())@b2_sparse

        #calculate attention laplacians using E function

        #Z0

        l0_sparse_1 = self.E_f(z0, self.W, self.k, l0_sparse,self.att_l0_1, self.dropout)
        l0_sparse_2 = self.E_f(z0, self.A, self.k, l0_sparse,self.att_l0_2, self.dropout)


        #Z1

        l1_sparse_1 = self.E_f(z1, self.W, self.k, l1_d_sparse,self.att_l1_1, self.dropout)
        l1_sparse_2 = self.E_f(z0, self.A, self.k, l1_d_sparse,self.att_l1_2, self.dropout,b1_sparse,t=True)
        l1_sparse_3 = self.E_f(z1, self.R, self.k, l1_u_sparse,self.att_l1_3, self.dropout)
        l1_sparse_4 = self.E_f(z2, self.D, self.k, l1_u_sparse,self.att_l1_4, self.dropout,b2_sparse)

        #Z2

        l2_sparse_1 = self.E_f(z2, self.R, self.k, l2_sparse,self.att_l2_1, self.dropout)
        l2_sparse_2 = self.E_f(z1, self.D, self.k, l2_sparse,self.att_l2_2, self.dropout,b2_sparse,t=True)


        z0_prime = self.compute_z0(z0, z1, b1_sparse, l0_sparse_1, l0_sparse_2,l0_sparse)
        z1_prime = self.compute_z1(z0, z1, z2, b1_sparse, b2_sparse, l1_sparse_1, l1_sparse_2, l1_sparse_3, l1_sparse_4,l1_sparse)
        z2_prime = self.compute_z2(z1, z2, b2_sparse, l2_sparse_1,l2_sparse_2,l2_sparse)

        return z0_prime, z1_prime, z2_prime

###########################################################################


# Combine SARNN (Aime & Damas)
# =======================================

#define SARNN
class SAN_RNN(pl.LightningModule):
    def __init__(self, in_size, hid_dim, out_size, k_len, drop_out,alpha_leakyrelu):
        super().__init__()
        self.k = k_len
        self.drop_out = drop_out
        self.alpha_leakyrelu = alpha_leakyrelu
        self.l0 = SAN(in_size, hid_dim,k=k_len,dropout=drop_out,alpha_leaky_relu=alpha_leakyrelu)
        self.l1 = SAN(hid_dim, hid_dim,k=k_len,dropout=drop_out,alpha_leaky_relu=alpha_leakyrelu)
        self.l2 = SAN(hid_dim, hid_dim,k=k_len,dropout=drop_out,alpha_leaky_relu=alpha_leakyrelu)
        # mlp 3 layers with relu nonlinear
        self.rnn = nn.RNN(input_size=3*hid_dim, hidden_size=hid_dim, num_layers=1, nonlinearity='tanh', batch_first=True)

        self.mlp1 = torch.nn.Linear(hid_dim, hid_dim)

        self.classifier = torch.nn.Linear(hid_dim,out_size)
    def forward(self, batch):
        
        z0 = batch.z0
        #print(z0.shape)
        z0 = z0.float()
        z0 = z0.reshape(z0.shape[0],1)
        #print(z0.shape)
        z1 = batch.z1
        z1 = z1.float()
        z1 = z1.reshape(z1.shape[0],1)
        z2 = batch.z2
        z2 = z2.float()
        z2 = z2.reshape(z2.shape[0],1)
        b1 = (batch.b1_index, batch.b1_val)
        b2 = (batch.b2_index, batch.b2_val)
        tri_index = batch.tri_idx
        

        n_nodes = z0.shape[0]
        n_edges = z1.shape[0]
        n_triangles = z2.shape[0]
        b1_sparse = torch.sparse_coo_tensor(b1[0], b1[1], size=(n_nodes, n_edges))
        b2_sparse = torch.sparse_coo_tensor(b2[0], b2[1], size=(n_edges, n_triangles))
        edge_batch = (b1_sparse.abs().t())@batch.batch.float()
        edge_batch_norm = (b1_sparse.abs().t())@torch.ones_like(batch.batch).float()
        edge_batch /= edge_batch_norm
        triangle_batch = (b2_sparse.abs().t())@edge_batch
        triangle_batch_norm = (b2_sparse.abs().t())@torch.ones_like(edge_batch).float()
        triangle_batch /= triangle_batch_norm



        z0_prime, z1_prime, z2_prime = self.l0(z0, z1, z2, b1, b2)
        #apply leaky relu to each z0_prime, z1_prime, z2_prime
        z0_prime = torch.nn.functional.leaky_relu(z0_prime, self.alpha_leakyrelu)
        z1_prime = torch.nn.functional.leaky_relu(z1_prime, self.alpha_leakyrelu)
        z2_prime = torch.nn.functional.leaky_relu(z2_prime, self.alpha_leakyrelu)

        #batchnorm

        z0_prime_1, z1_prime_1, z2_prime_1 = self.l1(z0_prime, z1_prime, z2_prime, b1, b2)
        #apply leaky relu to each z0_prime_1, z1_prime_1, z2_prime_1
        z0_prime_1 = torch.nn.functional.leaky_relu(z0_prime_1, self.alpha_leakyrelu)
        z1_prime_1 = torch.nn.functional.leaky_relu(z1_prime_1, self.alpha_leakyrelu)
        z2_prime_1 = torch.nn.functional.leaky_relu(z2_prime_1, self.alpha_leakyrelu)

        #sccnn node

        Z0 = z0_prime_1
        Z1 = z1_prime_1
        Z2 = z2_prime_1

        rows = []

        for l in tri_index[0]:
            #print(l)
            new = Z1[l,:]
            rows.append(new)
        
        out = torch.stack(rows)
        out = out.view(len(tri_index[0]), 1, Z1.shape[1]*3)
        out, _ = self.rnn(out)
		out = self.linear_out(out)[:,-1,:]

		return out
