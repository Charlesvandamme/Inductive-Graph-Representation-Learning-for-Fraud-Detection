# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:30:24 2020

@author: Hendrik
"""

import numpy as np
import networkx as nx
import pandas as pd
import scipy
from scipy.sparse import coo_matrix
import collections


class FIGRL():
    """ This class initializes the Fast Inductive Graph Representation Learning algorithm described in the paper by F. Jiang et al.
    ----------
    embedding_size : int
        The desired size of the resulting embeddings; together with the intermediate dimension it defines the approximation ratio
    intermediate_dimension : int
        The dimension of the matrix sketch M 
    Attributes
    ----------
    St : ndarray, shape (number of training nodes, intermediate dimension)
        Matrix of size # training nodes x intermediate dimension whose entries are independently drawn from N(0,1)
    V : ndarray, shape (final dimension, final dimension)
        The utter most right matrix of the singular value decomposition done on the normalized random walk matrix in the training step.
    sigma : ndarray, shape (final dimension, final dimension)
        The middle matrix of the singular value decomposition done on the normalized random walk matrix in the training step
    """
    def __init__(self, embedding_size, intermediate_dimension):
        self.embedding_size = embedding_size
        self.intermediate_dimension = intermediate_dimension
        self.St = None
        self.V = None
        self.sigma = None

    def fit(self, train_graph, S=None):
        """This function trains a figrl model.
        It returns the trained figrl model and a pandas datarame containing the embeddings generated for the train nodes.
        ----------
        train_graph : NetworkX Object
            The graph on which the training step is done on, containing only the seen training nodes.
        S : ndarray, shape (number of training nodes, intermediate dimension)
            A random matrix used to create the normalized random walk matrix; if non the fit definition creates a new one
        Returns
        -------
        figrl_train_emb : pandas Dataframe
            The embeddings created during the training step for the training nodes.
        """
        
        A = nx.adjacency_matrix(train_graph)
        n,m = A.shape
        diags = A.sum(axis=1).flatten()

        with scipy.errstate(divide='ignore'):
           diags_sqrt = 1.0/np.lib.scimath.sqrt(diags)
        diags_sqrt[np.isinf(diags_sqrt)] = 0
        DH = scipy.sparse.spdiags(diags_sqrt, [0], n, n, format='csr')

        Normalized_random_walk = DH.dot(A.dot(DH))
        if S is None:
            S = np.random.randn(n, self.intermediate_dimension) / np.sqrt(self.intermediate_dimension)
            #np.savetxt("S_train_matrix.csv", S, delimiter=",")

        C = Normalized_random_walk.dot(S)

        from scipy import sparse
        sC = sparse.csr_matrix(C)

        U, self.sigma, self.V = scipy.sparse.linalg.svds(sC, k=self.embedding_size, tol=0,which='LM')
        self.V = self.V.transpose()
        self.sigma = np.diag(self.sigma)
        
        figrl_train_emb = pd.DataFrame(U)
        figrl_train_emb = figrl_train_emb.set_index(figrl_train_emb.index)
        
        self.sigma = np.array(self.sigma)
        self.V = np.array(self.V)
        self.St = np.array(S)
        
        return figrl_train_emb

    def __create_inductive_dict(self, inductive_data ,list_connected_node_types):
        
        """
        This creates a collection of the inductive nodes as key and as values their interaction with other nodes in inductive step.
        
        Parameters
        ----------
        inductive_data : pandas Dataframe
            The row defines the incoming node, in the columns the different node types that can be connected to.
        list_connected_node_types : numpy list of pandas Dataframes
            The list contains the connected node collumns 
            
        Returns
        ----------
        inductive_dict: collection.OrderedDictionary
            The dictionay containing the inductive nodes as keys and the values their first degree neighbours
        """
        
        inductive_dict = {}
        for node in inductive_data.index:
            for i in list_connected_node_types:
                if i.loc[node] != None:
                    if node in inductive_dict:
                        inductive_dict[node].append(i.loc[node])  
                    else:
                        inductive_dict[node] = [i.loc[node]]
        inductive_dict = collections.OrderedDict(sorted(inductive_dict.items()))
        return inductive_dict
    
    def __get_vector(self, inductive_dict, train_degrees, max_id):
        """
        This creates the sparse vector_matrix used in the inductive step
        
        Parameters
        ----------
        inductive_dict :Ordered dictionary
            The dictionay containing the inductive nodes as keys and the values their first degree neighbours
        train_degrees : Ordered dictionary
            A dictionay containing the degrees of all the nodes in the inductive+train graph
        max_id: int
            The largest integer number used as ID 
        Returns
        ----------
        coo_matrix: scipy coo matrix
            This function returns a sparse coordination matrix with the normalized random walk vectors for the inductive nodes
        """
        row  = []
        col  = []
        data = []
        i = 0
        for node, v in inductive_dict.items():
            for n in v:
                if n is not None:
                    row.append(i)
                    col.append(n)
                    if n > max_id:
                        max_id = int(n)
                        #calculate value
                    inductive_degree = len([x for x in v if x != None])
                    value = 1/np.sqrt(inductive_degree)
                    value = value * (1/np.sqrt(train_degrees[n]))
                    data.append(value)
            i+=1
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        return coo_matrix((data, (row, col)), shape=(len(inductive_dict), max_id+1))

    def predict(self, graph, inductive_data, list_connected_node_types, maxid, inductive_index):
        """
        This function predicts embeddings for unseen nodes using a fitted figrl model.
        It returns the embeddings for these unseen nodes. 
        This function can inductively predict embeddings for useen nodes; the inductive part of Jiang et al.'s algorithm is slightly altered to improve the speed
        
        Parameters
        ----------
        graph : NetworkX Object
            The graph on which FIGRL is deployed (training + inductive nodes)
        inductive_data : pandas Dataframe
            The row defines the incoming node, in the columns the different node types that can be connected to.
        list_connected_node_types : numpy list of pandas Dataframes
            The list contains the connected node collumns 
        maxid: int
            The maximum integer ID for the training and inductive set
        inductive_index: RangeIndex
            The inductive indexes for the embeddings
        Returns
        ----------
        figrl_inductive_emb: pandas Dataframe
            The embeddings created during the training step for the inductive nodes.
        """
        inductive_dict = self.__create_inductive_dict(inductive_data, list_connected_node_types)
        
        degrees = nx.degree(graph)
        train_degrees = dict(degrees)
        train_degrees = collections.OrderedDict(sorted(train_degrees.items()))
        
        v = self.__get_vector(inductive_dict, train_degrees, maxid)

        S = np.random.randn(maxid+1, self.intermediate_dimension) / np.sqrt(self.intermediate_dimension)
        
        inductive_degrees = []

        for l in inductive_dict.values():
            x = 0
            for i in l:
                if i is not None:
                    x+=1
            inductive_degrees.append(x)
    

        sqrt_d_inv = np.array([1/np.sqrt(degree)  if degree > 0 else 0 for degree in inductive_degrees])
        sqrt_d_inv = scipy.sparse.spdiags(sqrt_d_inv,0, sqrt_d_inv.size, sqrt_d_inv.size)

        p = v.dot(S)
        U =(p.dot(self.V)).dot(np.linalg.inv(self.sigma))
        U = sqrt_d_inv.dot(U)
        
        figrl_inductive_emb = pd.DataFrame(U, index = inductive_index)
    
        return figrl_inductive_emb    

