# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:15:40 2020

@author: Charles
"""
from datetime import timedelta
import dateparser
import pandas as pd

class Timeframes:

    """
    This class initializes the a rolling window timeframe configuration.
    
    Parameters
    ----------
    date_column : Pandas dataframe
        A one column pandas dataframe containing the dates associated with records.
    step_size : int
        Step size (days) defines how many days are between the start days of two consecutive timeframes.
    window_size: int
        Defnes the window size (days) of a timeframe.
    """     
    
    
    
    def __init__(self, date_column, step_size=None, window_size=None):
        
        
        self.date_column = pd.DataFrame(date_column)
        self.step_size = step_size
        self.window_size = window_size
        
    
    def get_number_of_days(self):
        
        """
        This function returns the number of days covered by your data.
        The number of days can be used to identify a desired step and window size. 
        
        """
        
        date_column_name = list(self.date_column.columns)[0]
        return (max(self.date_column[date_column_name])-min(self.date_column[date_column_name])).days
    
    def get_number_of_timeframes(self):
        
        """
        This function returns the number of timeframes that can be created based on the configured step and window size. 
        
        """
        return int((1+ ((self.get_number_of_days()+1)-self.window_size)/self.step_size)//1)

    def train_inductive_split(self, data, hold_out_days):
        
        """
        This function returns two pandas dataframes: train data and inductive data.
        
        Parameters
        ----------
        data : Pandas dataframe
            The data that needs to be split
        hold_out_days : int
            The number of days that should be held out of the train set.

        """       
        if hold_out_days > self.window_size:
            raise ValueError("the number of hold out days cannot be larger than the total number of days in the window.")
            return
        date_column_name = list(self.date_column.columns)[0]
        timeframe_data = self.date_column.loc[data.index]
        end_date = dateparser.parse(max(timeframe_data[date_column_name]).strftime('%Y-%m-%d'))+timedelta(1)
        train_indices = (timeframe_data[timeframe_data[date_column_name] < (end_date-timedelta(hold_out_days))]).index
        inductive_indices = (timeframe_data[timeframe_data[date_column_name] >= (end_date-timedelta(hold_out_days))]).index

        return data.loc[train_indices], data.loc[inductive_indices]
    
    
    def get_timeframe_indices(self, timeframe):
        """
        This function returns the indices associated with a certain timeframe.
        
        Parameters
        ----------
        timeframe : int
            The numeric identifier of the timeframe for which the indices are requested. 

        """       
        date_column_name = list(self.date_column.columns)[0]
        self.date_column[date_column_name] = pd.to_datetime(self.date_column[date_column_name])
        start_date = dateparser.parse(min(self.date_column[date_column_name]).strftime('%Y-%m-%d'))
        st = (timeframe-1)*self.step_size
        timeframe_records = self.date_column[self.date_column[date_column_name] >= start_date +timedelta(st)]
        timeframe_records = timeframe_records[timeframe_records[date_column_name] < (start_date + (timedelta(self.window_size+st)))]

        
        return timeframe_records.index  
       