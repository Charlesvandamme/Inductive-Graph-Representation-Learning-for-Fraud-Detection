# Inductive Graph Representation Learning for Fraud Detection
This repository contains the code used in the experimental setup of our paper 'Inductive Graph Representation Learning for Fraud Detection'. The notebook 'Experimental Pipeline' contains the exact code used in our experiments, while the demo notebooks demonstrate a simplified version. For the latter, we load a small, artificially generated transaction network and run our pipeline via (i) the 'GraphSAGE' branch and (ii) the FI-GRL branch (see Figure below). Note that for the FI-GRL notebook, you need to configure your matlab.engine appropriately (see FI-GRL section below).

The implementation of our components can easily be extended with other functions to satisfy requirements from other researchers. We've modularised our code into the different pipeline components in the demo folder to encourage others to play with them. The credit card transaction network could for example easily be replaced by other networks to observe the predictive quality of GraphSAGE and FI-GRL embeddings. Our setup also allows to select any machine learning classifier, provided that a Python implementation is available. Lastly, other inductive graph representation learners can easily be plugged into our setup. 

# Installation

All required components for the demo notebooks can be installed via pip:
`pip install inductiveGRL`

Alternatively you can build this repository from source by cloning this repository
and subsequently running `pip install .` inside the cloned repo folder. 

# Abstract
Graphs can be seen as a universal language to describe and model a diverse set of complex systems and data structures. However, efficiently extracting topological information from dynamic graphs is not a straightforward task. Previous works have explored a variety of inductive graph representation learning frameworks, but despite the surge in development, little research deployed these techniques for real-life applications. Most earlier studies are restricted to a set of benchmark experiments, rendering their practical generalisability questionable. Our paper evaluates the proclaimed predictive performance of state-of-the-art inductive graph representation learning algorithms on highly imbalanced credit card transaction networks. More specifically, we assess the inductive capability of GraphSAGE and Fast Inductive Graph Representation Learning in a fraud detection setting. Credit card transaction fraud networks pose two crucial challenges for graph representation learners: First, these networks are highly dynamic, continuously encountering new transactions. Second, they are heavily imbalanced, with only a small fraction of transactions labeled as fraudulent. Our paper contributes to the literature by (i) proving how inductive graph representation learning techniques can be leveraged to enhance predictive performance for fraud detection and (ii) demonstrating the benefit of graph-level undersampling for representation learning in imbalanced networks.

# Experimental Pipeline
<img src="https://github.com/Charlesvandamme/Inductive-Graph-Representation-Learning-for-Fraud-Detection/blob/master/Figures/experimental_pipeline.JPG?raw=true"/>

### 1. Transaction Data ###
Any dataset that can be transformed into a graph can be used in our experimental setup. For our research, we used a real-life dataset to construct credit card transaction networks containing millions of transactions. This dataset includes information on the following features: anonymized identification of clients and merchants, merchant category code, country, monetary amount, time, acceptance, and fraud label. This real-life dataset is highly imbalanced and contains only 0.65% fraudulent transactions. Note that the demo data in this repository is artificaly generated for demonstration purposes. The `Timeframes` component derives the different timeframes for a rolling window setup given a step and window size.  

### 2. Graph Construction ###
The `GraphConstruction` component constructs the graphs that will be used by graph representation learners (e.g. FI-GRL and GraphSAGE) to learn node embeddings. We designed the credit card transaction networks as heterogeneous tripartite graphs containing client, merchant and transaction nodes. Because of this tripartite setup, representations can be learned for the transaction nodes. Only the transaction nodes are configured with node features.

### 3. GraphSAGE ###

The `HinSAGE` code deploys a supervised, heterogeneous implementation of the GraphSAGE framework called HinSAGE, to learn embeddings of the transaction nodes in the aforementioned graphs. 

### 4. FI-GRL ###
The `FIGRL` code learns embeddings of the transaction nodes in the aforementioned graphs using the Fast Inductive Graph Representation Learning Framework. We call the Matlab implementation of FI-GRL from our Jupyter notebooks, which requires an appropriate installation of matlab.engine in the same folder as the notebooks. If you wish to run FI-GRL from Python, please run the following command in Matlab:

`cd (fullfile(matlabroot,'extern','engines','python'))`\
`system('python setup.py install')`

This will generate a folder in matlabroot\extern\engines\python\build\lib called 'matlab' please copy this folder and place it on the same location as the notebook from which you want to call matlab.engine. If you don't know your matlab root, running 'matlabroot' in Matlab will return the appropriate path.

### 5. Classifier ###
The penultimate component in our pipeline uses the transaction node embeddings to classify the transaction nodes as fraudulent or legitimate. We chose to rely on XGBoost as a classification model, but other classifiers can easily be implemented. 

### 6. Evaluation ###
The `Evaluation` component contains functions for the Lift score, Lift curve and precision-recall curve. We focused on these evaluation metrics given the highly imbalanced nature of our dataset. However, this code can easily be extended to contain other evaluation metrics such as ROC plots. 
