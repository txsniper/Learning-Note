import pandas as pd 
import numpy as np 
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.utils import shuffle
import xgboost as xgb
import seaborn as sns 
from scipy import stats

pd.set_option('display.max_columns', None)

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

    def print_value_counts(self, all_data):
        keys = all_data.columns.values.tolist()
        for feature_name in keys:
            print('----------------' + feature_name + '-----------------')
            print(all_data[feature_name].value_counts())

    def data_analyze(self):
        all_data = pd.concat((self.train_data.drop('SalePrice', axis=1), self.test_data))
        #print(all_data.head(10))

        # 检查特征间的相关系数 
        train_corr = self.train_data.drop('Id',axis=1).corr()
        #print(train_corr['SalePrice'])
        a = plt.subplots(figsize=(40, 30))
        a = sns.heatmap(train_corr, vmax=0.8, square=True)# annot=True)
        plt.show()


        # 检查缺失值
        # isnull对每个元素判断是否为空, 用 True/False 填充结果矩阵
        # isnull().sum() 获取为空的行数
        # isnull().count() 获取矩阵总行数
        total = all_data.isnull().sum().sort_values(ascending=False)
        percent = (all_data.isnull().sum() / all_data.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total','Lost Percent'])
        #print(missing_data)
        #print(missing_data[missing_data.isnull().values==False].sort_values('Total', axis=0, ascending=False).head(20))

        #self.print_value_counts(all_data)



    def process_data(self):
        all_X = pd.concat((
            self.train_data.loc[:, 'MSSubClass':'SaleCondition'],
            self.test_data.loc[:, 'MSSubClass':'SaleCondition'])
        )

        # 删掉缺失值太多的特征
        missing_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
        all_X.drop(missing_features, axis=1, inplace=True)

        # 删除相同值太多的特征
        same_value_featrues = ['Street', 'LandSlope', 'LandContour', 'Condition2', 
            'RoofMatl', 'Heating', 'CentralAir', 'LowQualFinSF', 'BsmtHalfBath', 
            'KitchenAbvGr', 'Functional', '3SsnPorch', 'PoolArea', 'MiscVal'
        ]
        #all_X.drop(same_value_featrues, axis=1, inplace=True)

        # 删掉无用特征
        #print("delete useless feature")
        #all_X.drop(['Utilities'], axis=1, inplace=True)

        # 创造新的特征
        all_X['house_remod'] = all_X['YearRemodAdd'] - all_X['YearBuilt']
        all_X['room_area'] = all_X['TotRmsAbvGrd'] / all_X['GrLivArea']
        all_X['fu_room'] = all_X['FullBath'] / all_X['TotRmsAbvGrd']
        all_X['gr_room'] = all_X['BedroomAbvGr'] / all_X['TotRmsAbvGrd']

        # 填充缺失值
        na_col = all_X.dtypes[all_X.isnull().any()]
        for col in na_col.index:
            if na_col[col] != 'object':
                med = all_X[col].median()
                all_X[col].fillna(med, inplace=True)
            else:
                mode = all_X[col].mode()[0]
                all_X[col].fillna(mode, inplace=True)

        # 数值特征标准化
        print("numeric feature processing")
        numeric_feats = all_X.dtypes[all_X.dtypes != "object"].index
        all_X[numeric_feats] = all_X[numeric_feats].apply(lambda x: ((x - x.mean()) / x.std()))
        #all_X[numeric_feats] = all_X[numeric_feats].apply(lambda x: (x - x.mean() / x.std()))

        # 类别数据转换为数值数据
        print("categorary feature processing")
        all_X = pd.get_dummies(all_X, dummy_na=True)

        
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
        rf = RandomForestRegressor(n_estimators=80, random_state=14, max_depth=12)
        rf.fit(X_train, y_train)
        cross_score = np.sqrt(-cross_val_score(rf, X_train, y_train,n_jobs=4, cv=5, scoring='neg_mean_squared_error', verbose=1))
        print(cross_score)

    

    def gbr(self):
        X_train,  X_test, y_train = self.process_data()
        gbr_model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=1000, max_depth=5, random_state=15, loss='huber')
        gbr_model.fit(X_train, y_train)
        cross_score = np.sqrt(-cross_val_score(gbr_model, X_train, y_train,n_jobs=3, cv=5, scoring='neg_mean_squared_error', verbose=1))
        print(cross_score)
        predictions = gbr_model.predict(X_test)
        #print(predictions)
        self.write_predictions_2_csv(self.test_data, predictions, "gbr.csv")

    def rmsle_cv(self, model, X, y):
        n_folds = 5
        kf = KFold(n_folds, shuffle=True, random_state=41).get_n_splits(train.values)
        rmse = np.sqrt(-cross_val_score(model, X.values, y, scoring='neg_mean_squared_error', cv=kf))
        return rmse

    def curr_best(self):
        X_train,  X_test, y_train = self.process_data()
        xgb_model = xgb.XGBRegressor(
            colsample_bytree=0.5,
            gamma=0,
            learning_rate=0.05,
            max_depth=4,
            n_estimators=3000,
            min_child_weight=1.5,
            reg_alpha=0.6,
            reg_lambda=0.8,
            subsample=0.6,
            random_state=8,
        )
        xgb_model.fit(X_train, y_train)
        cross_score = np.sqrt(-cross_val_score(xgb_model, X_train, y_train,n_jobs=3, cv=5, scoring='neg_mean_squared_error', verbose=1))
        print(cross_score)
        predictions = xgb_model.predict(X_test)
        #print(predictions)
        self.write_predictions_2_csv(self.test_data, predictions, "xgb.csv")
    
    def xgb_grid_search(self):
        X_train,  X_test, y_train = self.process_data()
        xgb_model = xgb.XGBRegressor(
            colsample_bytree=0.5,
            gamma=0,
            learning_rate=0.05,
            max_depth=4,
            n_estimators=3000,
            min_child_weight=1.5,
            reg_alpha=0.6,
            reg_lambda=0.8,
            subsample=0.6,
            random_state=8,
        )
        xgb_model.fit(X_train, y_train)
        cross_score = np.sqrt(-cross_val_score(xgb_model, X_train, y_train,n_jobs=3, cv=5, scoring='neg_mean_squared_error', verbose=1))
        print(cross_score)
        predictions = xgb_model.predict(X_test)
        #print(predictions)
        self.write_predictions_2_csv(self.test_data, predictions, "xgb.csv")


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

from mxnet import ndarray as nd 
from mxnet import autograd
from mxnet import gluon
from mxnet import initializer
import matplotlib.pyplot as plt

square_loss = gluon.loss.L2Loss()

class MXNetSolution(object):
    def __init__(self, dir_name, train_file, test_file):
        self.solution = Solution(dir_name, train_file, test_file)

    def process(self):
        self.solution.load_data()
        X_train, X_test, y_train = self.solution.process_data()
        #print(X_train[0:5])
        #exit(0)
        num_train = X_train.shape[0]
        print(X_train.shape)
        print(y_train.shape)
        
        X_train = nd.array(X_train)
        y_train = nd.array(y_train)
        y_train.reshape((num_train, 1))
        X_test = nd.array(X_test)
        k = 3
        epochs = 80
        verbose_epoch = 75
        learning_rate = 0.005
        weight_decay = 0.9

        test = pd.read_csv(self.solution.test_file)
        train_loss, test_loss = self.k_fold_cross_valid(k, epochs, verbose_epoch, X_train, y_train, learning_rate, weight_decay)
        print("%d-fold validation: Avg train loss: %f, Avg test loss: %f" %(k, train_loss, test_loss))
        #self.learn(epochs, verbose_epoch, X_train, y_train, X_test, test, learning_rate, weight_decay)

    def get_net(self):
        net = gluon.nn.Sequential()
        drop_prob = 0.2
        # name_scope给参数一个唯一的名字，便于load/save模型
        with net.name_scope():
            #net.add(gluon.nn.Dense(100, activation='relu'))
            net.add(gluon.nn.Dense(70, activation='relu'))
            net.add(gluon.nn.Dense(30, activation='relu'))
            #net.add(gluon.nn.BatchNorm(axis=1))
            #net.add(gluon.nn.Dropout(drop_prob))
            #net.add(gluon.nn.Activation(activation='relu'))
            net.add(gluon.nn.Dense(1))
        net.initialize(init=initializer.Uniform())
        return net

    def get_rmse_log(self, net, X_train, y_train):
        num_train = X_train.shape[0]
        clipped_preds = nd.clip(net(X_train), 1, float('inf'))
        return np.sqrt(2 * nd.sum(square_loss(
            nd.log(clipped_preds), nd.log(y_train))).asscalar() / num_train)

    def train(self, net, X_train, y_train, X_test, y_test, epochs,
              verbose_epoch, learning_rate, weight_decay):
        train_loss = []
        if X_test is not None:
            test_loss = []
        batch_size = 100
        dataset_train = gluon.data.ArrayDataset(X_train, y_train)
        data_iter_train = gluon.data.DataLoader(
            dataset_train, batch_size, shuffle=True
        )
        trainer = gluon.Trainer(net.collect_params(), 'adam',
            {'learning_rate': learning_rate, 'wd': weight_decay}
        )
        net.collect_params().initialize(force_reinit=True)
        for epoch in range(epochs):
            for data, label in data_iter_train:
                with autograd.record():
                    output = net(data)
                    loss = square_loss(output, label)
                loss.backward()

                # 更新模型参数
                trainer.step(batch_size)

                cur_train_loss = self.get_rmse_log(net, X_train, y_train)
            if epoch > verbose_epoch:
                print("Epoch %d, train loss: %f" % (epoch, cur_train_loss))
            train_loss.append(cur_train_loss)
            if X_test is not None:
                cur_test_loss = self.get_rmse_log(net, X_test, y_test)
                test_loss.append(cur_test_loss)
        plt.plot(train_loss)
        plt.legend(['train'])
        if X_test is not None:
            plt.plot(test_loss)
            plt.legend(['train','test'])
        plt.show()
        if X_test is not None:
            return cur_train_loss, cur_test_loss
        else:
            return cur_train_loss

    def k_fold_cross_valid(self, k, epochs, verbose_epoch, X_train, y_train,
                       learning_rate, weight_decay):
        assert k > 1
        fold_size = X_train.shape[0] // k
        train_loss_sum = 0.0
        test_loss_sum = 0.0
        for test_i in range(k):
            X_val_test = X_train[test_i * fold_size: (test_i + 1) * fold_size, :]
            y_val_test = y_train[test_i * fold_size: (test_i + 1) * fold_size]

            val_train_defined = False
            for i in range(k):
                if i != test_i:
                    X_cur_fold = X_train[i * fold_size: (i + 1) * fold_size, :]
                    y_cur_fold = y_train[i * fold_size: (i + 1) * fold_size]
                    if not val_train_defined:
                        X_val_train = X_cur_fold
                        y_val_train = y_cur_fold
                        val_train_defined = True
                    else:
                        X_val_train = nd.concat(X_val_train, X_cur_fold, dim=0)
                        y_val_train = nd.concat(y_val_train, y_cur_fold, dim=0)
            net = self.get_net()
            train_loss, test_loss = self.train(
                net, X_val_train, y_val_train, X_val_test, y_val_test,
                epochs, verbose_epoch, learning_rate, weight_decay)
            train_loss_sum += train_loss
            print("Test loss: %f" % test_loss)
            test_loss_sum += test_loss
        return train_loss_sum / k, test_loss_sum / k    

    def learn(self, epochs, verbose_epoch, X_train, y_train, X_test, test, learning_rate,
          weight_decay):
        net = self.get_net()
        self.train(net, X_train, y_train, None, None, epochs, verbose_epoch,
                learning_rate, weight_decay)
        preds = net(X_test).asnumpy()
        test['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
        submission = pd.concat([test['Id'], test['SalePrice']], axis=1)
        submission.to_csv(self.solution.dir_name + '/submission.csv', index=False)    
    

if __name__ == "__main__":
    dir_name = os.path.dirname(os.path.realpath(__file__))
    obj = Solution(dir_name, dir_name + '/train.csv', dir_name + '/test.csv')
    obj.load_data()
    obj.xgb_grid_search()
    #obj.random_forest()
    #obj.gbr()
    #obj.data_analyze()

    #net_obj = MXNetSolution(dir_name, dir_name + '/train.csv', dir_name + '/test.csv')
    #net_obj.process()
