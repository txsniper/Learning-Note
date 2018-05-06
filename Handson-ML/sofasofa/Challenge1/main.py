import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import os
class Solution(object):
    def __init__(self, curr_dir, train_file, test_file):
        self.curr_dir = curr_dir
        self.train_file = train_file
        self.test_file = test_file

    def load_data(self):
        self.train_data = pd.read_csv(self.train_file)
        self.test_data = pd.read_csv(self.test_file)

    def describe(self):
        self.train_data.info()
        self.test_data.info()
        ret = self.train_data.describe()
        print(ret)

    def preprocessing_data(self, data):
        city_no = pd.get_dummies(data['city'], prefix='city')
        data = pd.concat([data, city_no], axis=1)
        data = data.drop(['city'], axis=1) 

        weather_no = pd.get_dummies(data['weather'], prefix='weather')
        data = pd.concat([data, weather_no], axis=1)
        data = data.drop(['weather'], axis=1)

        workday_no = pd.get_dummies(data['is_workday'], prefix='is_workday')
        data = pd.concat([data, workday_no], axis=1)
        data = data.drop(['is_workday'], axis=1)

        # 增加时间范围属性
        data['hour_cat'] = 0
        data.loc[((data['hour'] > 20) & (data['hour'] <= 24)) | ((data['hour'] >= 0) & (data['hour'] <= 6 )), 'hour_cat'] = 1
        data.loc[((data['hour'] >= 7) & (data['hour'] <= 10)), 'hour_cat'] = 2
        data.loc[((data['hour'] >= 11) & (data['hour'] <= 13)), 'hour_cat'] = 3
        data.loc[((data['hour'] >= 14) & (data['hour'] <= 16)), 'hour_cat'] = 4
        data.loc[((data['hour'] >= 17) & (data['hour'] <= 20)), 'hour_cat'] = 5
        
        hour_no = pd.get_dummies(data['hour_cat'], prefix='hour_cat')
        data = pd.concat([data, hour_no], axis=1)
        data = data.drop(['hour_cat'], axis=1)
        return data     

    def write_predictions_2_csv(self, test_data,  predictions, csv_name):
        result = pd.DataFrame({'id':test_data['id'].as_matrix(), 'y':predictions.astype(np.int32)})
        result.to_csv(self.curr_dir + "/" + csv_name, index=False)

    def grand_boost(self):
        train_data = self.preprocessing_data(self.train_data)
        y = train_data['y']
        train_data = train_data.drop(['id', 'y'], axis=1)
        print(train_data.head())
        # 归一化
        scaler = StandardScaler()
        temp1_scaler_param = scaler.fit(train_data['temp_1'].values.reshape(-1, 1))
        train_data['temp_1_scaled'] = scaler.fit_transform(train_data['temp_1'].values.reshape(-1, 1), temp1_scaler_param)
        temp2_scaler_param = scaler.fit(train_data['temp_2'].values.reshape(-1, 1))
        train_data['temp_2_scaled'] = scaler.fit_transform(train_data['temp_2'].values.reshape(-1, 1), temp2_scaler_param)

        train_data.drop(['temp_1', 'temp_2'], axis=1, inplace=True)
        print(train_data.head())
        X = train_data

        param_grid = {
            'n_estimators' : [170, 180, 200],
            'max_depth' : [4, 5, 6],
            'learning_rate' : [0.075, 0.08, 0.082, 0.084, 0.09]
        }
        gb = GradientBoostingRegressor()
        '''
        lin_gre = LinearRegression()
        param_grid = {
            'normalize' : [True, False],
        }
        '''
        #gb.fit(X.as_matrix(), y.as_matrix())

        grid = GridSearchCV(gb, param_grid,cv=5, scoring='neg_mean_squared_error', verbose=1)
        #grid = GridSearchCV(lin_gre, param_grid, cv=10, scoring='neg_mean_squared_error', verbose=1)
        grid.fit(X, y)
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
        bgb = grid.best_estimator_
        test_data = self.preprocessing_data(self.test_data)
        test = test_data.drop(['id'], axis=1)
        test['temp_1_scaled'] = scaler.fit_transform(test['temp_1'].values.reshape(-1, 1), temp1_scaler_param)
        test['temp_2_scaled'] = scaler.fit_transform(test['temp_2'].values.reshape(-1, 1), temp2_scaler_param)
        test.drop(['temp_1', 'temp_2'], axis=1, inplace=True)
        print(test.head())
        predictions = bgb.predict(test.as_matrix())
        predictions_pd = pd.DataFrame(predictions, columns=['y'])
        predictions_pd.loc[predictions_pd['y'] < 0, 'y'] = 0
        print(predictions_pd.head())
        #predictions = gb.predict(test.as_matrix())
        predictions = predictions_pd['y'].values
        print(predictions[0:5])
        self.write_predictions_2_csv(test_data, predictions, "grand_boost.csv")
        



if __name__ == "__main__":
    dir_name = os.path.dirname(os.path.realpath(__file__))
    obj = Solution(dir_name, dir_name + '/train.csv', dir_name + '/test.csv')
    obj.load_data()
    #obj.describe()
    obj.grand_boost()
