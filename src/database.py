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

class GraphData():
    '''
    This is a class for the Genrated Graph Dataset
    
    Attributes:
        num (int)= The number of graphs required for each type
        nodes (int)= The number of nodes required in each graph
    '''
    def __init__(self, num, nodes, m_list = None, p_list = None):
        '''
        The constructor for GraphData Class
        
        Parameters:
            num = The number of graphs required for each type
            nodes = The number of nodes required in each graph       
        '''
        self.num = num
        self.nodes = nodes
        if(m_list == None):
            self.m_list = [1.0, nodes/8.0, nodes/4.0, 3.0*nodes/8.0, nodes/2.0]
        else:
            self.m_list = m_list
        if(p_list == None):
            self.p_list = np.linspace(1.0/num, num/2.0, 4)
        else:
            self.p_list = p_list
        
        self.graphs = []
        self.graphadjs = []
        self.graphlabels = []        
        self.tokenisedgraphlabels = []
        self.moments = {}

    def generate_BAGraphs(self):
        '''
        Generates n=num number of graphs of type BA. Also add labels "BA".
        '''
        num_each_group = int(self.num/len(self.m_list))
        for m in self.m_list:
            for i in range(num_each_group):
                graph = nx.barabasi_albert_graph(self.nodes, int(m))
                self.graphs.append(graph)
                self.graphlabels.append("BA")
                  
    def generate_ERGraphs(self):
        '''
        Generates n=num number of graphs of type ER. Also add labels "ER"
        '''
        num_each_group = int(self.num/len(self.p_list))
        
        for p in self.p_list:
            for i in range(num_each_group):
                graph = nx.erdos_renyi_graph(self.nodes, p, directed=False)
                self.graphs.append(graph)
                self.graphlabels.append("ER")
                
    def generate_configurational_graphs(self):
        '''
        Generate n=num of graphs of type configurational from the graphs already built. Also add labels "CG"
        '''
        temp_graphs = self.graphs.copy()
        for graph in temp_graphs:
            graph_con = configuration_model_1(np.array(nx.adjacency_matrix(graph).todense()))
            graph_con = nx.Graph(graph_con)
            self.graphs.append(graph_con)
            self.graphlabels.append("CG")            
                
    def generate_adjs(self):
        '''
        Generate adjacent matrix from the generated graphs.
        '''
        for graph in self.graphs:
            self.graphadjs.append(np.array(nx.adjacency_matrix(graph).todense()).astype(np.float32))
     
    def tokenise(self, map_dict):
        '''
        Generate tokenised graph labels from the labels for training purposes.
        '''
        self.tokenisedgraphlabels = list(map(map_dict.get, self.graphlabels))

    def generate_moments(self, moment):
        Poweradj = self.graphadjs.copy()
        for i in range(len(Poweradj)):
            for _ in range(moment-1):
                Poweradj[i] = Poweradj[i]*self.graphadjs[i]
        self.moments[moment] = np.array(Poweradj).sum(-1)
