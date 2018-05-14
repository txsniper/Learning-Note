import pandas as pd 
import numpy as np 
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score


class Solution(object):
    def __init__(self, dir_name, train_file, test_file):
        self.dir_name = dir_name
        self.train_file = train_file
        self.test_file = test_file

    def load_data(self):
        self.train_data = pd.read_csv(self.train_file)
        self.test_data = pd.read_csv(self.test_file)
        ret = self.train_data.describe()
        #print(self.train_data.head())

    

    def process_data(self):
        all_X = pd.concat((
            self.train_data.loc[:, 'MSSubClass':'SaleCondition'],
            self.test_data.loc[:, 'MSSubClass':'SaleCondition'])
        )

        # 删掉无用特征
        print("delete useless feature")
        all_X.drop(['Utilities'], axis=1, inplace=True)

        # 数值特征标准化
        print("numeric feature processing")
        numeric_feats = all_X.dtypes[all_X.dtypes != "object"].index
        all_X[numeric_feats] = all_X[numeric_feats].apply(lambda x: (x - x.mean() / x.std()))

        # 类别数据转换为数值数据
        print("categorary feature processing")
        all_X = pd.get_dummies(all_X, dummy_na=True)

        # 平均值填充缺失值
        all_X = all_X.fillna(all_X.mean())

        #print(all_X.head())
        #return all_X
        num_train = self.train_data.shape[0]
        X_train = all_X[:num_train].as_matrix()
        X_test  = all_X[num_train:].as_matrix()
        y_train = self.train_data.SalePrice.as_matrix()
        return X_train, X_test, y_train


    def write_predictions_2_csv(self, test_data,  predictions, csv_name):
        result = pd.DataFrame({'Id':test_data['Id'].as_matrix(), 'SalePrice':predictions})
        result.to_csv(self.dir_name + "/" + csv_name, index=False)

    def random_forest(self):
        X_train,  X_test, y_train = self.process_data()
        rf = RandomForestRegressor(n_estimators=600, random_state=14, max_depth=15)
        #rf.fit(X_train, y_train)
        cross_score = np.sqrt(-cross_val_score(rf, X_train, y_train,n_jobs=4, cv=5, scoring='neg_mean_squared_error', verbose=1))
        print(cross_score)

    def gbr(self):
        X_train,  X_test, y_train = self.process_data()
        gbr_model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=150, max_depth=4, random_state=15)
        gbr_model.fit(X_train, y_train)
        cross_score = np.sqrt(-cross_val_score(gbr_model, X_train, y_train,n_jobs=4, cv=5, scoring='neg_mean_squared_error', verbose=1))
        print(cross_score)
        predictions = gbr_model.predict(X_test)
        print(predictions)
        self.write_predictions_2_csv(self.test_data, predictions, "gbr.csv")

    def random_forest_grid_search(self):
        X_train,  X_test, y_train = self.process_data()
        param_grid = {
            'n_estimators' : [ 100, 130, 150, 180, 200, 300],
            'max_depth' : [3, 4, 5, 6, 7, 8],
            'max_leaf_nodes' : [2, 3, 4, 5],
        }
        scores = ['neg_mean_squared_error']
        bclf = None
        bclf = None
        for score in scores:
            rf = RandomForestRegressor(random_state=14)
            grid = GridSearchCV(rf, param_grid, n_jobs=4, cv=5, scoring='%s' % score, verbose=1)
            #grid = RandomizedSearchCV(rf, param_grid, cv=10, scoring='%s' % score, verbose=1)
            grid.fit(X_train, y_train)
        
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
            bclf.fit(X, y)  
            y_true = y  
            y_pred = bclf.predict(X)  
            y_pred_pro = bclf.predict_proba(X)  
            y_scores = pd.DataFrame(y_pred_pro, columns=bclf.classes_.tolist())[1].values  
            print(classification_report(y_true, y_pred))  
            auc_value = roc_auc_score(y_true, y_scores)
            '''
        predictions = bclf.predict(X_test)
        #self.write_predictions_2_csv(test_data, predictions, "random_forest.csv")


if __name__ == "__main__":
    dir_name = os.path.dirname(os.path.realpath(__file__))
    obj = Solution(dir_name, dir_name + '/train.csv', dir_name + '/test.csv')
    obj.load_data()
    #obj.random_forest()
    obj.gbr()