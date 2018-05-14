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
        num_train = X_train.shape[0]
        print(X_train.shape)
        print(y_train.shape)
        
        X_train = nd.array(X_train)
        y_train = nd.array(y_train)
        y_train.reshape((num_train, 1))
        X_test = nd.array(X_test)
        k = 5
        epochs = 800
        verbose_epoch = 780
        learning_rate = 5
        weight_decay = 0.1

        train_loss, test_loss = self.k_fold_cross_valid(k, epochs, verbose_epoch, X_train, y_train, learning_rate, weight_decay)
        print("%d-fold validation: Avg train loss: %f, Avg test loss: %f" %(k, train_loss, test_loss))

    def get_net(self):
        net = gluon.nn.Sequential()
        # name_scope给参数一个唯一的名字，便于load/save模型
        with net.name_scope():
            #net.add(gluon.nn.Dense(100, activation='relu'))
            #net.add(gluon.nn.Dense(50, activation='relu'))
            net.add(gluon.nn.Dense(1))
        net.initialize(init=initializer.Xavier())
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

    def learn(self, epochs, verbose_epoch, X_train, y_train, test, learning_rate,
          weight_decay):
        net = get_net()
        train(net, X_train, y_train, None, None, epochs, verbose_epoch,
                learning_rate, weight_decay)
        preds = net(X_test).asnumpy()
        test['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
        submission = pd.concat([test['Id'], test['SalePrice']], axis=1)
        submission.to_csv(self.dir_name + '/submission.csv', index=False)    
    

if __name__ == "__main__":
    dir_name = os.path.dirname(os.path.realpath(__file__))
    obj = Solution(dir_name, dir_name + '/train.csv', dir_name + '/test.csv')
    obj.load_data()
    #obj.random_forest()
    #obj.gbr()

    net_obj = MXNetSolution(dir_name, dir_name + '/train.csv', dir_name + '/test.csv')
    net_obj.process()