import pandas as pd
import numpy as np

np.set_printoptions(threshold=100000)
pd.set_option('display.max_columns', 500)
class Solution(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.train_csv = data_path + "/train"
        self.test_csv = data_path + "/test"
        self.all_features = ['click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']

    def display_feature(self, data_frame, feature_name):
        print("----------" + feature_name + "------------")
        ret = data_frame[feature_name].count()
        print(ret)
        ret1 = data_frame[feature_name].value_counts().nlargest(20)
        print(ret1)

    def load_data(self):
        self.train_data = pd.read_csv(self.train_csv, nrows=20000000)
        #print(self.train_data.columns.values.tolist())
        #print(self.train_data.describe())
        #print(self.train_data.values_count())
        print(self.train_data.describe())
        for feature in self.all_features:
            self.display_feature(self.train_data, feature)


def main():
    path = "/home/tanxing/kaggle/avazu-ctr-prediction"
    solution = Solution(path)
    solution.load_data()

if __name__ == "__main__":
    main()
