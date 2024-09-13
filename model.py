import torch
from torch.nn import Linear as Lin

import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from layers import HGPSLPool,HGPSLPoolsp
#from torch_geometric.nn import ChebConv 
from opt import * 
import math 
from GCN import GCN,GCNsp
from PAE import PAE
#from sparse_softmax import Sparsemax
from torch_geometric.utils import dense_to_sparse
from dknn import dknn
##########################################################################################################################


class Mybraingnn(torch.nn.Module):
    def __init__(self, dim1, dim2, num_features, pooling_ratio,nhid,dropout_ratio):
        super(Mybraingnn, self).__init__()
        #parameters in network
        K=3
        bias = False
        self.in_dim1 = dim1
        self.out_dim1 = dim2
        self.in_dim2 = dim2
        self.out_dim2 = dim2
        self.num_features = num_features
        self.pooling_ratio = pooling_ratio
        self.sample = True
        self.sparse = True
        self.sl = False
        self.lamb = 1.0
        self.nhid = nhid
        self.dropout_ratio = dropout_ratio
        self.dim3 = math.ceil(pooling_ratio*math.ceil(pooling_ratio*111))
        self.dim4 = (self.dim3+2)*self.out_dim2
        
        #layers in network 
        self.relu = torch.nn.ReLU(inplace=True)
        self.gcn1 = GCN(self.in_dim1, self.out_dim1)
        self.pool1 = HGPSLPool(self.num_features, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.gcn2 = GCN(self.in_dim2, self.out_dim2)
        self.pool2 = HGPSLPool(self.num_features, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)        
        
        self.lin1 = torch.nn.Linear(self.dim4, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.bn = torch.nn.BatchNorm1d(self.nhid)
        self.lin3 = torch.nn.Linear(self.nhid, 1)
    def forward(self,data,device): 
        x, edge_index = data.x, data.edge_index
        edge_attr = None
        batch = data.batch
        
        x = self.relu(self.gcn1(x, edge_index, edge_attr, batch))
        x, edge_index, edge_attr,batch = self.pool1(x, edge_index, edge_attr,batch)
        x1 = torch.cat([gmp(x,batch), gap(x,batch)], dim=1)

        x = self.relu(self.gcn2(x, edge_index, edge_attr, batch))
        x, edge_index, edge_attr,batch = self.pool2(x, edge_index,edge_attr,batch)
        x2 = torch.cat([gmp(x,batch), gap(x,batch)], dim=1)
        
        x_1 = x.view(data.num_graphs, self.dim3 , self.out_dim2)
        x = x_1.view(data.num_graphs,self.dim3*self.out_dim2)
        x = F.relu(x)
        x3 = F.relu(x1) + F.relu(x2) 
        x = torch.cat([x,x3],dim=1)

        features = F.relu(self.lin1(x))
        features = self.bn(features)
       
        x_lo = torch.flatten(self.lin3(features))

        return x_lo

class dualbraingnn(torch.nn.Module):
    def __init__(self, dim1, dim2, num_features, pooling_ratio,nhid,dropout_ratio):
        super(dualbraingnn, self).__init__()
        #parameters in network
        K=3
        bias = False
        self.in_dim1 = dim1
        self.out_dim1 = dim2
        self.in_dim2 = dim2
        self.out_dim2 = dim2
        self.num_features = num_features
        self.pooling_ratio = pooling_ratio
        self.sample = True
        self.sparse = True
        self.sl = False
        self.lamb = 1.0
        self.nhid = nhid
        self.dropout_ratio = dropout_ratio
        self.dim3 = math.ceil(pooling_ratio*math.ceil(pooling_ratio*111))
        self.dim4 = (self.dim3+2)*self.out_dim2
        
        #layers in network 
        self.relu = torch.nn.ReLU(inplace=True)
        #self.gcn1 = ChebConv(self.in_dim1, self.out_dim1, K, normalization='sym', bias=bias)
        self.gcn1 = GCN(self.in_dim1, self.out_dim1)
        self.pool1 = HGPSLPool(self.num_features, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        #self.gcn2 = ChebConv(self.in_dim2, self.out_dim2, K, normalization='sym', bias=bias)
        self.gcn2 = GCN(self.in_dim2, self.out_dim2)
        self.pool2 = HGPSLPool(self.num_features, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)        
        
        self.gcnsp1 = GCNsp(self.in_dim1, self.out_dim1)
        self.poolsp1 = HGPSLPoolsp(self.num_features, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        #self.gcn2 = ChebConv(self.in_dim2, self.out_dim2, K, normalization='sym', bias=bias)
        self.gcnsp2 = GCNsp(self.in_dim2, self.out_dim2)
        self.poolsp2 = HGPSLPoolsp(self.num_features, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        self.lin1 = torch.nn.Linear(self.dim4*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.bn = torch.nn.BatchNorm1d(self.nhid)
        self.lin3 = torch.nn.Linear(self.nhid, 1)
    def forward(self,data,device): 
        x, edge_index = data.x, data.edge_index
        edge_attr = None
        batch = data.batch
        
        #smoothing GCN
        xm = self.relu(self.gcn1(x, edge_index, edge_attr, batch))
        xm, edge_index, edge_attr,batch = self.pool1(xm, edge_index, edge_attr,batch)
        xm1 = torch.cat([gmp(xm,batch), gap(xm,batch)], dim=1)

        xm = self.relu(self.gcn2(xm, edge_index, edge_attr, batch))
        xm, edge_index, edge_attr,batch = self.pool2(xm, edge_index,edge_attr,batch)
        xm2 = torch.cat([gmp(xm,batch), gap(xm,batch)], dim=1)
        
        x_1 = xm.view(data.num_graphs, self.dim3 , self.out_dim2)
        xm = x_1.view(data.num_graphs,self.dim3*self.out_dim2)
        xm = F.relu(xm)
        xm3 = F.relu(xm1) + F.relu(xm2) 
        xm = torch.cat([xm,xm3],dim=1)
        
        #sharpning GCN
        x, edge_index = data.x, data.edge_index
        edge_attr = None
        batch = data.batch

        xp = self.relu(self.gcnsp1(x, edge_index, edge_attr, batch))
        xp, edge_index, edge_attr,batch = self.poolsp1(xp, edge_index, edge_attr,batch)
        xp1 = torch.cat([gmp(xp,batch), gap(xp,batch)], dim=1)

        xp = self.relu(self.gcnsp2(xp, edge_index, edge_attr, batch))
        xp, edge_index, edge_attr,batch = self.poolsp2(xp, edge_index,edge_attr,batch)
        xp2 = torch.cat([gmp(xp,batch), gap(xp,batch)], dim=1)
        
        x_2 = xp.view(data.num_graphs, self.dim3 , self.out_dim2)
        xp = x_2.view(data.num_graphs,self.dim3*self.out_dim2)
        xp = F.relu(xp)
        xp3 = F.relu(xp1) + F.relu(xp2) 
        xp = torch.cat([xp,xp3],dim=1)
        
        features = F.relu(self.lin1(x))
        features =  F.relu(self.lin2(features))
        x_lo=F.softmax(features)
        return x_lo,features

class ddbraingnn(torch.nn.Module):
    def __init__(self, dim1, dim2, num_features, pooling_ratio,nhid,dropout_ratio):# indim, ratio, nclass, k=8, R=200):
        super(ddbraingnn, self).__init__()
        #parameters in network
        K=3
        bias = False
        self.in_dim1 = dim1
        self.out_dim1 = dim2
        self.in_dim2 = dim2
        self.out_dim2 = dim2
        self.num_features = num_features
        self.pooling_ratio = pooling_ratio
        self.sample = True
        self.sparse = True
        self.sl = False
        self.lamb = 1.0
        self.nhid = nhid
        self.dropout_ratio = dropout_ratio
        self.dim3 = math.ceil(pooling_ratio*math.ceil(pooling_ratio*111))
        self.dim4 = (self.dim3+2)*self.out_dim2*2
       
        #layers in network 
        self.relu = torch.nn.ReLU(inplace=True)
        #self.gcn1 = ChebConv(self.in_dim1, self.out_dim1, K, normalization='sym', bias=bias)
        self.gcn1 = GCN(self.in_dim1, self.out_dim1)
        self.pool1 = HGPSLPool(self.out_dim1*2, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        #self.gcn2 = ChebConv(self.in_dim2, self.out_dim2, K, normalization='sym', bias=bias)
        self.gcn2 = GCN(self.in_dim2*2, self.out_dim2)
        self.pool2 = HGPSLPool(self.out_dim2*2, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)        
        
        self.gcnsp1 = GCNsp(self.in_dim1, self.out_dim1)
        self.gcnsp2 = GCNsp(self.in_dim2*2, self.out_dim2)
       
        self.lin1 = torch.nn.Linear(self.dim4, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.bn = torch.nn.BatchNorm1d(self.nhid)
        self.lin3 = torch.nn.Linear(self.nhid, 1)
    def forward(self,data,device): 
        x, edge_index = data.x, data.edge_index
        edge_attr = None
        batch = data.batch
        
        #dual convolution layer 1
        xm = self.relu(self.gcn1(x, edge_index, edge_attr, batch))
        xp = self.relu(self.gcnsp1(x, edge_index, edge_attr, batch))
        x =torch.cat([xm,xp],dim=1)
        
        #pooling layer 1
        x, edge_index, edge_attr,batch = self.pool1(x, edge_index, edge_attr,batch)
        x1 = torch.cat([gmp(x,batch), gap(x,batch)], dim=1)

        # dual convolution layer 2
        xm = self.relu(self.gcn2(x, edge_index, edge_attr, batch))
        xp = self.relu(self.gcnsp2(x, edge_index, edge_attr, batch))
        x = torch.cat([xm,xp],dim=1)
        
        #pooling layer 2
        x, edge_index, edge_attr,batch = self.pool2(x, edge_index,edge_attr,batch)
        x2 = torch.cat([gmp(x,batch), gap(x,batch)], dim=1)
        
        x_1 = x.view(data.num_graphs, self.dim3 , self.out_dim2*2)
        x = x_1.view(data.num_graphs,self.dim3*self.out_dim2*2)
        x = torch.cat([x,x1,x2],dim=1)
        
        features = F.relu(self.lin1(x))
        features =  F.relu(self.lin2(features))
        x_lo=F.softmax(features)
        return x_lo,features  
class psdbraingnn(torch.nn.Module):
    def __init__(self, dim1, dim2, num_features, pooling_ratio,nhid,dropout_ratio, dropout):# indim, ratio, nclass, k=8, R=200):
        super(psdbraingnn, self).__init__()
        #parameters in network
        K=3
        bias = False
        self.in_dim1 = dim1
        self.out_dim1 = dim2
        self.in_dim2 = dim2
        self.out_dim2 = dim2
        self.num_features = num_features
        self.pooling_ratio = pooling_ratio
        self.sample = True
        self.sparse = True
        self.sl = True
        self.lamb = 1.0
        self.nhid = nhid
        self.dropout_ratio = dropout_ratio
        self.dim3 = math.ceil(pooling_ratio*math.ceil(pooling_ratio*111))
        self.dim4 = (self.dim3+2)*self.out_dim2*2
        self.dropout=dropout
    
        self.relu = torch.nn.ReLU(inplace=True)
        self.edge_net = PAE(input_dim=self.in_dim1, dropout=self.dropout)

        self.sparse_knn=dknn(self.in_dim1)

        self.gcn1 = GCN(self.in_dim1, self.out_dim1)
        self.pool1 = HGPSLPool(self.out_dim1*2, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.gcn2 = GCN(self.in_dim2*2, self.out_dim2)
        self.pool2 = HGPSLPool(self.out_dim2*2, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)        
        
        self.gcnsp1 = GCNsp(self.in_dim1, self.out_dim1)
        self.gcnsp2 = GCNsp(self.in_dim2*2, self.out_dim2)
       
        self.lin1 = torch.nn.Linear(self.dim4, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.bn = torch.nn.BatchNorm1d(self.nhid)
        self.lin3 = torch.nn.Linear(self.nhid, 1)
    def forward(self,data,device): 
        x, edge_index,edge_attr,batch = data.x, data.edge_index,data.edge_attr,data.batch
    
        edge_attr = torch.squeeze(self.edge_net(edge_attr))
      
        edge_attr = self.sparse_knn(x,edge_attr, edge_index,device)
        row,col=edge_index
        adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=device)
        adj[row, col] = edge_attr
        edge_index, edge_attr = dense_to_sparse(adj)
        del adj,row,col

        xm = self.relu(self.gcn1(x, edge_index, edge_attr, batch))
        xp = self.relu(self.gcnsp1(x, edge_index, edge_attr, batch))
        x =torch.cat([xm,xp],dim=1)

        x, edge_index, edge_attr,batch = self.pool1(x, edge_index, edge_attr,batch)
        x1 = torch.cat([gmp(x,batch), gap(x,batch)], dim=1)

        xm = self.relu(self.gcn2(x, edge_index, edge_attr, batch))
        xp = self.relu(self.gcnsp2(x, edge_index, edge_attr, batch))
        x = torch.cat([xm,xp],dim=1)
        
        x, edge_index, edge_attr,batch = self.pool2(x, edge_index,edge_attr,batch)
        x2 = torch.cat([gmp(x,batch), gap(x,batch)], dim=1)
        
        x_1 = x.view(data.num_graphs, self.dim3 , self.out_dim2*2)
        x = x_1.view(data.num_graphs,self.dim3*self.out_dim2*2)
        x = torch.cat([x,x1,x2],dim=1)
        
        features = F.relu(self.lin1(x))
        features =  F.relu(self.lin2(features))
        x_lo=F.softmax(features)
        return x_lo,features



