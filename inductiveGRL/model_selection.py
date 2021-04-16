from datetime import timedelta
import dateparser
import pandas as pd
from pandas.arrays import DatetimeArray

class TimeBasedFold:

    def __init__(self, date_column, step_size=None, train_size=None, test_size=None):
        
        if not isinstance(step_size, timedelta):
            raise ValueError('step_size should be of type datetime.timedelta')

        if not isinstance(train_size, timedelta):
            raise ValueError('train_size should be of type datetime.timedelta')

        if not isinstance(test_size, timedelta):
            raise ValueError('test_size should be of type datetime.timedelta')

        if not isinstance(date_column, DatetimeArray):
            try:
                date_column = DatetimeArray(datecolumn)
            except:
                print("The provided date column cannot be parsed. Please provide a datetime column in pandas DatetimeArray format.")

        self.date_column = date_column
        self.step_size = step_size
        self.train_size = train_size
        self.test_size = test_size
        self.fold_size = self.train_size + self.test_size

    def split(self): 

        train_indices = []
        test_indices = []
        first_timestamp = min(self.date_column)
        last_timestamp = max(self.date_column)
        current_timestamp = first_timestamp
        while current_timestamp < last_timestamp - self.fold_size:
            start_train = current_timestamp
            end_train = current_timestamp + self.train_size
            start_test = end_train
            end_test = start_test + self.test_size

            train_indices.append(((self.date_column >= start_train) & (self.date_column < end_train)).nonzero()[0])
            test_indices.append(((self.date_column >= start_test) & (self.date_column < end_test)).nonzero()[0])

            current_timestamp = current_timestamp + self.step_size

        return train_indices, test_indices
    


