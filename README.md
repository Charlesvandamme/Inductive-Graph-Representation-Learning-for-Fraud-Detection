# Inductive Graph Representation Learning for Fraud Detection
This repository contains the code used in the experimental setup of our paper 'Inductive Graph Representation Learning for Fraud Detection. The notebook 'Experimental Pipeline' contains the exact code used in our experiments, while 'Demo Pipeline' demonstrates a simplified version. For the latter, we load a small sample transaction network (artificially generated) and run our pipeline via the 'GraphSAGE' branch (see Figure below). 

The implementation of our components can easily be extended with other functions to satisfy requirements from other researchers. We've modularised our code into the different pipeline components to encourage others to play with them. The credit card transaction network could for example easily be replaced by other networks to observe the predictive quality of GraphSAGE and FI-GRL embeddings. Our setup also allows to select any machine learning classifier, provided that a Python implementation is available. Lastly, other inductive graph representation learners can easily be plugged into our setup.   

# Abstract
Graphs can be seen as a universal language to describe and model a diverse set of complex systems and data structures. However, efficiently extracting topological information from dynamic graphs is not a straightforward task. Previous works have explored a variety of inductive graph representation learning frameworks, but despite the surge in development, little research deployed these techniques for real-life applications. Most earlier studies are restricted to a set of benchmark experiments, rendering their practical generalisability questionable. Our paper evaluates the proclaimed predictive performance of state-of-the-art inductive graph representation learning algorithms on highly imbalanced credit card transaction networks. More specifically, we assess the inductive capability of GraphSAGE and Fast Inductive Graph Representation Learning in a fraud detection setting. Credit card transaction fraud networks pose two crucial challenges for graph representation learners: First, these networks are highly dynamic, continuously encountering new transactions. Second, they are heavily imbalanced, with only a small fraction of transactions labeled as fraudulent. Our paper contributes to the literature by (i) proving how inductive graph representation learning techniques can be leveraged to enhance predictive performance for fraud detection and (ii) demonstrating the benefit of graph-level undersampling for representation learning in imbalanced networks.

# Experimental Pipeline
<img src="https://github.com/Charlesvandamme/Inductive-Graph-Representation-Learning-for-Fraud-Detection/blob/master/Figures/experimental_pipeline.JPG?raw=true"/>

### 1. Transaction Data ###
Any dataset that can be transformed into a graph can be used in our experimental setup. For our research, we used a real-life dataset to construct credit card transaction networks containing millions of transactions. This dataset includes information on the following features: anonymized identification of clients and merchants, merchant category code, country, monetary amount, time, acceptance, and fraud label. This real-life dataset is highly imbalanced and contains only 0.65% fraudulent transactions. Note that the demo data in this repository is an anonymized version.

### 3. Graph Construction ###
The `Graph Construction` code constructs the graphs that will be used by graph representation learners (e.g. FI-GRL and GraphSAGE) to learn node embeddings. We designed the credit card transaction networks as heterogeneous tripartite graphs containing client, merchant and transaction nodes. Because of this tripartite setup, representations can be learned for the transaction nodes. Only the transaction nodes are configured with node features. 

### 4. GraphSAGE ###

The `GraphSAGE` code deploys a supervised, heterogeneous implementation of the GraphSAGE framework, to learn embeddings of the transaction nodes in the aforementioned graphs. 

### 5. FI-GRL ###
The `FI-GRL` code learns embeddings of the transaction nodes in the aforementioned graphs using the Fast Inductive Graph Representation Learning Framework.

### 6. Classifier ###
The penultimate component in our pipeline uses the transaction node embeddings to classify the transaction nodes as fraudulent or legitimate. We chose to rely on XGBoost as a classification model, but other classifiers can easily be implemented. 
