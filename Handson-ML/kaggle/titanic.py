import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class Titanic(object):
    def __init__(self, csv_train, csv_test):
        self.train_path =  csv_train
        self.test_path = csv_test

    def load_data(self):
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)

    def data_describe(self):
        self.train_data.info()
        ret = self.train_data.describe()
        print(ret)
        self.train_data.hist(bins=50, figsize=(20, 15))
        plt.show()
    
    def handle_missing_value(self):
        # age
        

if __name__ == "__main__":
    dir_name = os.path.dirname(os.path.realpath(__file__))
    print(dir_name)
    obj = Titanic(dir_name + '/train.csv', dir_name + '/test.csv')
    obj.load_data()
    obj.data_describe()
    