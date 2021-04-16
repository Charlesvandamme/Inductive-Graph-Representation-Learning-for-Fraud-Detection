# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:01:50 2020

@author: Charles

"""
import networkx as nx
import stellargraph as sg

class GraphConstruction:
    
    """
    This class initializes a networkX graph
    
     Parameters
    ----------
    nodes : dict(str, iterable)
        A dictionary with keys representing the node type, values representing
        an iterable container of nodes (list, dict, set etc.)
    edges : 2-tuples (u,v) or 3-tuples (u,v,d)
        Each edge given in the container will be added to the graph. 
    features: dict(str, (str/dict/list/Dataframe)
        A dictionary with keys representing node type, values representing the node
        data.      
    
    """
    
    g_nx = None
    node_features = None
    
    def __init__(self, nodes, edges, features = None):
        self.g_nx = nx.Graph()
        self.add_nodes(nodes)
        self.add_edges(edges)
        
        if features is not None:
            self.node_features = features
    
    def add_nodes(self, nodes):
        
        for key, values in nodes.items():
            self.g_nx.add_nodes_from(values, ntype=key)      
            
    def add_edges(self, edges):
        
        for edge in edges:
            self.g_nx.add_edges_from(edge)
            
    def get_stellargraph(self):
        return sg.StellarGraph(self.g_nx, node_type_name="ntype", node_features=self.node_features)
    
    def get_edgelist(self):
        edgelist = []
        for edge in nx.generate_edgelist(self.g_nx):
            edgelist.append(str(edge).strip('{}'))
        el = []
        for edge in edgelist:
            splitted = edge.split()
            numeric = map(float,splitted)
            el.append(list(numeric))
        return el
