# -*- coding: utf-8 -*-
"""
The model is defined in this file. 
@author : Humair Raj Khan
"""
import torch
import tqdm
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, TensorDataset, DataLoader

activation_dict = {
    'linear' : None,
    'relu' : F.relu
}

nxseed = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
np.random.seed(5)
torch.manual_seed(5)
if device is not "cpu":
    torch.cuda.manual_seed(5)

class GCN(nn.Module):
    '''
    A class to represent the GCN model
    Attributes :
        fc(Object of class nn.Linear) : Node wise dense layer
        activation(string) : Activation to be applied to the dense layer output
    '''
    def __init__(self, inputnodefeat, outputnodefeat, activation):
        '''Constructor for the class GCN
        Parameters:
            inputnodefeat(int) : Number of features in the input node
            outputnodefeat(int) : Number of features in the output node
            activation(string) : Activation to be used
        '''
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_features=inputnodefeat, out_features=outputnodefeat).to(device)
        self.activation = activation
    def forward(self, adj, h):
        '''The Forward pass of the function
        Parameters :
            adj(torch.tensor) : The adjoint of the graph.
            h(torch.tensor) : The feature matrix of the nodes of the graph. 
        Return:
            (torch.tensor) : The deep representation of the node level output of the graph.
        '''
        x = torch.bmm(adj, h)
        x = self.fc(x)
        if(self.activation != 'linear'):
            x = activation_dict[self.activation](x)
        return x
    
class GCNModules(nn.Module):
    '''
    According to the normalisation types normalise the adjoint and simultaneously make a gcn for each adjoint types
    and concatenate it to make node level features. This will be further completed by the GCNModularLayer to make
    the complete layer.
    Attribute:
        inputnodefeat(int) : The number of features for the input nodes.
        outputnodefeat(int) : The number of features required in the output nodes.
        activation(string) : The activation which is to be used.
        normalisation_types(list(string)) : The types of adjoint normalisation to be used
    '''
    def __init__(self, inputnodefeat, outputnodefeat, activation, normalisation_types):
        '''Constructor for the GCNModules Class'''
        super(GCNModules, self).__init__()
        self.inputnodefeat = inputnodefeat
        self.outputnodefeat = outputnodefeat
        self.activation = activation
        self.normalisation_types = normalisation_types
        self.gcnmodules = nn.ModuleList()
        for i in range(len(normalisation_types)):
            self.gcnmodules.append(GCN(self.inputnodefeat, self.outputnodefeat, self.activation).to(device))
    def forward(self, adj, hlist):
        '''The Forward pass of the function
        Parameters :
            adj(torch.tensor) : The adjoint of the graph.
            hlist(torch.tensor) : The concatenated feature matrix of the nodes of the graph. Each dimension will be used
                                for the each of the normalisation types.
        Return:
            (torch.tensor) : The concatenated deep representation of the node level output of the graph.
        '''
        hlist = torch.split(hlist, self.inputnodefeat, dim = -1)
        
        normalised_adjs = []
        if("a" in self.normalisation_types):
            normalised_adjs.append(adj)
        rowsum = adj.sum(1)
        r_inv = torch.pow(rowsum, -1)
        r_inv[r_inv == float("Inf")] = 0.
        d = torch.diag_embed(r_inv)
        
        if("da" in self.normalisation_types):
            da = d.bmm(adj)
            normalised_adjs.append(da)
        if("dad" in self.normalisation_types):
            d = torch.sqrt(d)
            dad = (d.bmm(adj)).bmm(d)
            normalised_adjs.append(dad)
            
        listofnodefeat = [gcn(adj, h) for gcn, adj, h in zip(self.gcnmodules, normalised_adjs, hlist)]
        return listofnodefeat

class GCNModularLayer(nn.Module):
    '''
    Take the output from the GCNModules and apply a dense layer to complete the layer.
    Attributes :
        inputnodefeat(int) : The size of the input node feature.
        unit(int) : The number of features required in the layer intermediate output. The output of the layer
                        is however finally converted back to the inputnodefeat size.
        activation(string) : The activation which is to be used.
        normalisation_types(list(string)) : The types of adjoint normalisation to be used
        gcnmodules(Object of class GCNModules) : The object of the GCN Modules which has to be completed into the layer.
        dense(Object of class nn.Linear) : The layer which will be applied over the gcnmodules object to complete the layer.
                        This will convert back the output to the inputnodefeature size.
    '''
    def __init__(self, inputnodefeat, unit, activation, normalisation_types):
        '''
        Constructor of the class
        '''
        super(GCNModularLayer, self).__init__()
        self.inputnodefeat = inputnodefeat
        self.unit = unit
        self.activation = activation
        self.normalisation_types = normalisation_types
        self.gcnmodules = GCNModules(self.inputnodefeat, self.unit, self.activation, self.normalisation_types).to(device)
        self.dense = nn.Linear(in_features=self.unit*len(self.normalisation_types), out_features=self.inputnodefeat*len(self.normalisation_types)).to(device)
    def forward(self, adj, hlist):
        '''The Forward pass of the function
        Parameters :
            adj(torch.tensor) : The adjoint of the graph.
            hlist(torch.tensor) : The concatenated feature matrix of the nodes of the graph. Each dimension will be used
                            for the each of the normalisation types.
        Return:
            (torch.tensor) : The deep representation of the node level output of the graph. This will be split to be fed 
                            into the next layer.
        '''
        intermediateoutput = self.gcnmodules(adj, hlist)
        intermediateoutput = torch.cat(intermediateoutput, axis = -1).to(device)
        intermediateoutput = self.dense(intermediateoutput)
        if(self.activation != 'linear'):
            intermediateoutput = activation_dict[self.activation](intermediateoutput)
        return intermediateoutput
    
# model = GCNModularLayer(2, F.relu)
# model(torch.from_numpy(X[0][np.newaxis]).type(torch.float32), torch.cat([torch.from_numpy(h[0][np.newaxis])]*3, axis = -1).type(torch.float32))

class ModularGCN(nn.Module):
    '''
    The Class which will construct all the layers for the Modular GCN.
    Attributes :
        inputnodefeat(int) : The length of the features of the input nodes. The intermediate layer feature are converted
                        back to this size by design. This can however be altered.
        outputnodefeat(int) : The final features of each node after passing through all the layers.
        unit(list(int)) : The number of features required in intermediate hidden layers. The size of this list
                        indicates the length of the model.
        activation(string) : The activation which is to be used for all the layers.
        skip(bool) : Weather skip connection is required at the last layer. This connection connects all the 
                        intermediate layers output to the final layer
        normalisation_types(list(string)) : The types of adjoint normalisation to be used in the intermediate
                        layers
        modulelayers(list(Objects of class GCN Modular Layer)) : List of layers of object type GCNModularLayer.
                        These are the layers present in the model. 
        dense(Object of class nn.Linear) : The layer which will produce the final output. 
    '''
    def __init__(self, inputnodefeat, outputnodefeat, units, activation, skip = True, normalisation_types = ["a", "da", "dad"]):
        '''Constructor of the class ModularGCN'''
        super(ModularGCN, self).__init__()
        self.inputnodefeat = inputnodefeat
        self.outputnodefeat = outputnodefeat
        self.units = units
        self.activation = activation
        self.skip = skip
        self.normalisation_types = normalisation_types
        self.modulelayers = nn.ModuleList()
        #Make instances of the type GCNModularLayer. These are equal to the number of elements in the units list.
        for unit in units:
            self.modulelayers.append(GCNModularLayer(self.inputnodefeat, unit, self.activation, self.normalisation_types).to(device))
        self.dense = nn.Linear(len(normalisation_types)*len(units), out_features=outputnodefeat).to(device)
        
    def forward(self, adj, h):
        '''Forward Propagation. Rules for the forward propagation
            1. The hlist is created from the h passed into the model. This is done for the different types of 
                normalisation methods to work on each set.
            2. The modulelayers are run each one at a time and the output is calculated.
            3. According to weather skip is true or false all the layers output are concatanated or the final
                output is used.
            4. Finally a dense layer is applied to the output to produce the required number of features 
                for each node in the graph.

        Parameters :
            adj(torch.tensor) : The adjoint matrix of the graphs. Size: [batch_size, nodes, nodes]
            h(torch.tensor) : The feature matrix of the nodes of the graphs. Size: [batch_size, nodes, features]
        Return :
            (torch.tensor) : The output of the model. Size: [batch_size, nodes, outputnodefeat]
        '''
        hlist = torch.cat([h]*len(self.normalisation_types), axis = -1)
        intermediateoutputs = []
        for modulelayer in self.modulelayers:
            if(len(intermediateoutputs)==0):
                intermediateoutputs.append(modulelayer(adj, hlist))
            else:
                intermediateoutputs.append(modulelayer(adj, intermediateoutputs[-1]))
        if(self.skip):
            intermediateoutputs = torch.cat(intermediateoutputs, axis = -1)
        else:
            intermediateoutputs = intermediateoutputs[-1]
        finaloutput = self.dense(intermediateoutputs)
        #Use LEAKYRELU in the final output.
        finaloutput = F.leaky_relu(finaloutput, negative_slope=.3)
        return finaloutput


if __name__ == "__main__":
    params = {"units" : [unit]*layer,
    "activation" : "linear",
    "outputnodefeat" : 10,
    "inputnodefeat" : 1, 
    "skip" : True}

    model = ModularGCN(params["inputnodefeat"], params["outputnodefeat"], params["units"], params["activation"], params["skip"]).to(device)
    ADJ_sample = np.array([[[0., 1., 0., 1.],
                            [1., 0., 1., 1.],
                            [0., 1., 0., 1.],
                            [1., 1., 1., 0.]],

                           [[0., 1., 1., 0.],
                            [1., 0., 0., 1.],
                            [1., 0., 0., 1.],
                            [0., 1., 1., 0.]]])
    h_sample = np.array([[[1.],
                        [1.],
                        [1.],
                        [1.]],

                       [[1.],
                        [1.],
                        [1.],
                        [1.]]])
    outputnodefeatures = model(torch.from_numpy(ADJ_sample).type(torch.float32),torch.from_numpy(h_sample).type(torch.float32))
    print("The output node features are :", outputnodefeatures)
