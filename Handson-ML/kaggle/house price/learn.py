import numpy as np 
import pandas as pd 
import xgboost as xgb 

class Solution(object):
    def __init__(self, dir_name, train_file, test_file):
        self.dir_name = dir_name
        self.train_file = train_file
        self.test_file = test_file

    def load_data(self):
        self.train_data = pd.read_csv(self.train_file)
        #self.train_data = shuffle(self.train_data)
        self.test_data = pd.read_csv(self.test_file)
        #ret = self.train_data.describe()
        #print(self.train_data.head())
    

    def process_data(self):
        