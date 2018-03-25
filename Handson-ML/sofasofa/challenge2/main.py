import pandas as pd 
import numpy as np 
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import os

class Main():
    def __init__(self, curr_dir, csv_train, csv_test):
        self.curr_dir = curr_dir
        self.train_path =  csv_train
        self.test_path = csv_test

    def load_data(self):
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)

    def write_predictions_2_csv(self, test_data,  predictions, csv_name):
        result = pd.DataFrame({'CaseId':test_data['CaseId'].as_matrix(), 'Evaluation':predictions.astype(np.float32)})
        result.to_csv(self.curr_dir + "/" + csv_name, index=False)

    def random_forest(self):
        self.load_data()
        train_data = self.train_data
        train_data = train_data.drop(['CaseId'], axis=1)
        y = train_data['Evaluation']
        X = train_data.drop(['Evaluation'], axis=1)
        rf = RandomForestClassifier(random_state=42, n_estimators=2000)
        
        #rf = RandomForestClassifier(n_estimators=200, max_depth=4, max_leaf_nodes=5, random_state=42)
        scores = cross_val_score(rf, X, y, cv=8, scoring='roc_auc', verbose=1, n_jobs=5)
        rf.fit(X, y)
        print(scores)
        '''
        param_grid = {
            'n_estimators' : [100, 120, 180, 190, 200, 220],
            'max_depth' : [3, 4, 5, 6],
            'max_leaf_nodes' : [2, 3, 4, 5],
        }
        grid = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', verbose=1, n_jobs=4)
        grid.fit(X, y)
        #rf.fit(X, y)


        print("Best parameters set found on development set:")  
        print()  
        print(grid.best_params_)  
        print()  
        print("Grid scores on development set:")  
        print()  
        means = grid.cv_results_['mean_test_score']  
        stds = grid.cv_results_['std_test_score']  
        for mean, std, params in zip(means, stds, grid.cv_results_['params']):  
            print("%0.3f (+/-%0.03f) for %r"  % (mean, std * 2, params))  
        print()  
        print("Detailed classification report:")  
        print()  
        print("The model is trained on the full development set.")  
        print("The scores are computed on the full evaluation set.")  
        print()
        bclf = grid.best_estimator_
        '''
        test_data = self.test_data
        test = test_data.drop(['CaseId'], axis=1)
        #predictions = bclf.predict_proba(test)
        predictions = rf.predict_proba(test)
        # 每个样本标签为1的概率
        predictions_1 = predictions[:,1]
        print(predictions_1[0:5])
        self.write_predictions_2_csv(test_data, predictions_1, "random_forest.csv")

if __name__ == "__main__":
    dir_name = os.path.dirname(os.path.realpath(__file__))
    obj = Main(dir_name, dir_name + '/train.csv', dir_name + '/test.csv')
    obj.random_forest()