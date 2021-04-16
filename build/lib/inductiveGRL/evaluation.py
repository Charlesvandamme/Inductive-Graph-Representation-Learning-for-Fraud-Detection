# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:07:24 2020

@author: Charles

"""

import numpy as np
import pandas as pd
import scikitplot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from matplotlib import pyplot

class Evaluation:

    """
    This class initializes the evaluation of a classification model.
    
    Parameters
    ----------
    probabilities : iterable
        The predicted probabilities per class for the classification model.
    labels : iterable
        The labels corresponding with the predicted probabilities.    
    name : str
        The name of the used configuration
    """    
    
    def __init__(self, probabilities, labels, name):
        
        self.probabilities = probabilities
        self.labels = labels
        self.name = name
        
    def lift_score(self, percentile):
     
        """
        This function calculates the lift score for a given percentile.
        It returns the Lift score for that percentage.
        
        Parameters
        ----------
        percentile : float
            Specifies the percentile for which the Lift score should be calculated
         
        """       
        prob_class_1 = list(self.probabilities[:,1])
        prob_class_1_with_labels = np.column_stack((prob_class_1, self.labels))
        sorted_df = pd.DataFrame(prob_class_1_with_labels).sort_values(by=[0], ascending=[False]).head(round((len(prob_class_1_with_labels)*percentile)))
        percent_class_1_with_model = (sorted_df[1].value_counts()[1]/(sorted_df[1].value_counts()[0]+sorted_df[1].value_counts()[1]))
        percent_without_model = (self.labels.value_counts()[1]/(self.labels.value_counts()[0]+self.labels.value_counts()[1]))
        lift_score = percent_class_1_with_model/percent_without_model
  
        print('The ', percentile*100, "% Lift is equal to: ", lift_score)
      
        return lift_score
        
    def lift_curve(self):
        """
        This function plots the Lift curve.
        
        """
        scikitplot.metrics.plot_lift_curve(self.labels, self.probabilities)
        pyplot.show()
        
        

        
    def pr_curve(self):

        """
        This function plots the precision recall curve for the used classification model and a majority classifier.
        
        """
        probs = self.probabilities[:, 1]
        precision, recall, _ = precision_recall_curve(self.labels, probs)
        #no_skill = (self.labels.value_counts()[1]/(self.labels.value_counts()[0]+self.labels.value_counts()[1]))
        #pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Majority classifier')
        pyplot.plot(recall, precision, label=self.name)
        # axis labels
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        # show the legend
        pyplot.legend()
        # show the plot
        
        print('Average precision-recall score XGBoost: {0:0.10f}'.format(average_precision_score(self.labels, probs)))
