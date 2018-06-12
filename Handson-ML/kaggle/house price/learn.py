import numpy as np 
import pandas as pd 
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.learning_curve import validation_curve
import lightgbm as lgb
import os

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
    
    def fit(self, X, y):
        self.models_ = [ clone(x) for x in self.models ]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


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
    

    #特渣工程之瞎搞特征,别问我思路是什么,纯属乱拍脑袋搞出来,而且对结果貌似也仅有一点点影响
    '''
    data['house_remod']:  重新装修的年份与房建年份的差值
    data['livingRate']:   LotArea查了下是地块面积,这个特征是居住面积/地块面积*总体评价
    data['lot_area']:    LotFrontage是与房子相连的街道大小,现在想了下把GrLivArea换成LotArea会不会好点?
    data['room_area']:   房间数/居住面积
    data['fu_room']:    带有浴室的房间占总房间数的比例
    data['gr_room']:    卧室与房间数的占比
    '''
    def create_feature(self, data):
        #是否拥有地下室
        hBsmt_index = data.index[data['TotalBsmtSF']>0]
        data['HaveBsmt'] = 'NO'
        data.loc[hBsmt_index,'HaveBsmt'] = 'YES'
        data['house_remod'] = data['YearRemodAdd']-data['YearBuilt']
        data['livingRate'] = (data['GrLivArea']/data['LotArea'])*data['OverallCond']
        data['lot_area'] = data['LotFrontage']/data['GrLivArea']
        data['room_area'] = data['TotRmsAbvGrd']/data['GrLivArea']
        data['fu_room'] = data['FullBath']/data['TotRmsAbvGrd']
        data['gr_room'] = data['BedroomAbvGr']/data['TotRmsAbvGrd']
        return data

    def process_data(self):
        all_X = pd.concat((
            self.train_data.loc[:, 'MSSubClass':'SaleCondition'],
            self.test_data.loc[:, 'MSSubClass':'SaleCondition'])
        )

        # 构造新特征
        all_X = self.create_feature(all_X)

        # 删掉缺失值太多的特征
        missing_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
        all_X.drop(missing_features, axis=1, inplace=True)


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

        # 查看NaN数据
        #print(all_X[all_X['HaveBsmt'].isna()])
        self.feature_list = all_X.columns

        num_train = self.train_data.shape[0]
        X_train = all_X[:num_train].values
        X_test  = all_X[num_train:].values
        y_train = self.train_data.SalePrice.values

        #print(y_train)
        #exit(0)
        return X_train, X_test, y_train

    
    def rmsle_cv(self, model, X, y):
        n_folds = 5
        kf = KFold(n_folds, shuffle=True, random_state=41).get_n_splits()
        rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf, verbose=1, n_jobs=4))
        return rmse

    def rmsle(self, pred, y):
        return np.sqrt(np.square(np.log(pred+1) - np.log(y+1)).mean())

    def run(self):
        X_train, X_test, y_train = self.process_data()

        # 模型选择
        ## LASSO Regression :
        lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
        # Elastic Net Regression
        ENet = make_pipeline(
            RobustScaler(), ElasticNet(
            alpha=0.0005, l1_ratio=.9, random_state=3))
        # Kernel Ridge Regression
        KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
        ## Gradient Boosting Regression
        GBoost = GradientBoostingRegressor(
            n_estimators=3000,
            learning_rate=0.05,
            max_depth=4,
            max_features='sqrt',
            min_samples_leaf=15,
            min_samples_split=10,
            loss='huber',
            random_state=5)
        ## XGboost
        model_xgb = xgb.XGBRegressor(
            colsample_bytree=0.4603,
            gamma=0.0468,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=1.7817,
            n_estimators=2200,
            reg_alpha=0.4640,
            reg_lambda=0.8571,
            subsample=0.5213,
            silent=1,
            random_state=7,
            nthread=-1)
        ## lightGBM
        model_lgb = lgb.LGBMRegressor(
            objective='regression',
            num_leaves=5,
            learning_rate=0.05,
            n_estimators=720,
            max_bin=55,
            bagging_fraction=0.8,
            bagging_freq=5,
            feature_fraction=0.2319,
            feature_fraction_seed=9,
            bagging_seed=9,
            min_data_in_leaf=6,
            min_sum_hessian_in_leaf=11)
        ## 对这些基本模型进行打分

        score = self.rmsle_cv(lasso, X_train, y_train)
        print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
        score = self.rmsle_cv(ENet, X_train, y_train)
        print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
        score = self.rmsle_cv(KRR, X_train, y_train)
        print(
            "Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
        score = self.rmsle_cv(GBoost, X_train, y_train)
        print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(),
                                                                score.std()))
        score = self.rmsle_cv(model_xgb, X_train, y_train)
        print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
        score = self.rmsle_cv(model_lgb, X_train, y_train)
        print("LGBM score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

        # 几个模型融合
        stacked_averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))
        score = self.rmsle_cv(stacked_averaged_models, X_train, y_train)
        print("Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
        stacked_averaged_models.fit(X_train, y_train)
        stacked_train_pred = stacked_averaged_models.predict(X_train)
        #stacked_pred = np.expm1(stacked_averaged_models.predict(X_test))
        stacked_pred = stacked_averaged_models.predict(X_test)
        print(self.rmsle(stacked_train_pred, y_train))

        # xgboost
        model_xgb.fit(X_train, y_train)
        xgb_train_pred = model_xgb.predict(X_train)
        #xgb_pred = np.expm1(model_xgb.predict(X_test))
        xgb_pred = model_xgb.predict(X_test)
        print(self.rmsle(xgb_train_pred, y_train))
        
        # lightGBM
        model_lgb.fit(X_train, y_train)
        lgb_train_pred = model_lgb.predict(X_train)
        #lgb_pred = np.expm1(model_lgb.predict(X_test))
        lgb_pred = model_lgb.predict(X_test)
        print(self.rmsle(lgb_train_pred, y_train))
        '''RMSE on the entire Train data when averaging'''

        print('RMSLE score on train data:')

        # 融合方式: 加权平均
        print(self.rmsle(stacked_train_pred * 0.25 + xgb_train_pred * 0.3 +
                    lgb_train_pred * 0.45, y_train))

        # 模型融合的预测效果
        ensemble = stacked_pred * 0.25 + xgb_pred * 0.3 + lgb_pred * 0.45

        self.write_predictions_2_csv(self.test_data, ensemble, "ensemble.csv")

    def grid_search_cv(self, model, param, X_train, y_train):
        grid_model = GridSearchCV(estimator=model, 
            param_grid=param, 
            scoring='neg_mean_squared_error',
            cv=5,
            verbose=1,
            n_jobs=4
        )
        grid_model.fit(X_train, y_train)
        evalute_res = grid_model.cv_results_
        for mean_score, params in zip(evalute_res['mean_test_score'], evalute_res['params']):
            print(np.sqrt(-mean_score), params)
        
        print('best param: {0}'.format(grid_model.best_params_))
        print('best model score: {0}'.format(np.sqrt(-grid_model.best_score_)))

    def lightgbm_gird_search(self):
        X_train,  X_test, y_train = self.process_data()
        cv_params = {
            #'n_estimators' : [2000, 2500, 3000, 3500],
            #'bagging_fraction' : [0.4, 0.5, 0.6]
            #'feature_fraction' : [0.2, 0.3, 0.4, 0.5, 0.6]
            #'lambda_l1' : [0.5, 0.6, 0.7, 0.8, 0.9, 1]
            #'lambda_l2' : [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            #'learning_rate' : [0.03, 0.05, 0.07, 0.1, 0.12, 0.15]
            #'num_leaves' : [8, 12, 16, 24, 32]
            #'mean_data_in_leaf' : [4, 6, 8, 10]
            #'max_bin' : [40, 60, 80, 100, 150]
        }
        other_params = {
            'objective'     : 'regression',
            'num_leaves'    : 12,
            'learning_rate' : 0.05,
            'n_estimators'  : 3000,
            'max_bin'       : 60,
            'bagging_fraction' : 0.5,
            'bagging_freq'  : 5,
            'feature_fraction' : 0.3,
            'feature_fraction_seed' : 9,
            'bagging_seed'  : 9,
            'min_data_in_leaf'  : 6,
            'min_sum_hessian_in_leaf' : 11,
            'lambda_l1' : 0.7,
            'lambda_l2' : 0.4,
        }
        model_lgb = lgb.LGBMRegressor(
            **other_params
        )
        self.grid_search_cv(model_lgb, cv_params, X_train, y_train)
    
    def xgb_grid_search(self):
        X_train,  X_test, y_train = self.process_data()
        cv_params = {
            #'n_estimators': [2000, 2500, 3000],
            #'max_depth' : [3,4,5,6],
            #'learning_rate' : [0.03, 0.04, 0.05]
            #'min_child_weight' : [0.5, 1.0, 1.5, 2.0],
            #'subsample': [0.5, 0.6, 0.7],
            #'colsample_bytree': [0.4, 0.5, 0.6, 0.7],
            #'reg_alpha' : [0.5, 0.6, 0.7],
            #'reg_lambda' : [0.5, 0.6, 0.7, 0.8, 0.9],
            #'gamma' : [0.1, 0.2, 0.15],
            
        }
        other_params = {
            'learning_rate' : 0.05,
            'n_estimators' : 2000,
            'max_depth' : 4,
            'min_child_weight' : 1.5,
            'subsample' : 0.7,
            'colsample_bytree' : 0.6,
            'gamma' : 0,
            'reg_alpha' : 0.6,
            'reg_lambda' : 0.8,
        }
        xgb_model = xgb.XGBRegressor(**other_params)
        self.grid_search_cv(xgb_model, cv_params, X_train, y_train)

    def xgb_regressor(self):
        X_train,  X_test, y_train = self.process_data()
        xgb_model = xgb.XGBRegressor(
            colsample_bytree=0.4,
            gamma=0.1,
            learning_rate=0.05,
            max_depth=4,
            n_estimators=3000,
            min_child_weight=2,
            reg_alpha=0.6,
            reg_lambda=0.8,
            subsample=0.6,
            random_state=8,
        )
        #kf = KFold(5, shuffle=True, random_state=41).get_n_splits()
        #cross_score = np.sqrt(-cross_val_score(xgb_model, X_train, y_train,n_jobs=3, cv=kf, scoring='neg_mean_squared_error', verbose=1))

        score = self.rmsle_cv(xgb_model, X_train, y_train)
        print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
        xgb_model.fit(X_train, y_train)

        predictions = xgb_model.predict(X_test)
        important = xgb_model.feature_importances_
        feature_imp_dict = {}
        for feature_name, feature_imp in zip(self.feature_list, important):
            feature_imp_dict[feature_name] = feature_imp
        
        feature_imp_dict = sorted(feature_imp_dict.items(), key=lambda d:d[1], reverse=True)
        for key, value in feature_imp_dict:
            print(key + ": " + str(value))
    
        self.write_predictions_2_csv(self.test_data, predictions, "xgb.csv")

    def write_predictions_2_csv(self, test_data,  predictions, csv_name):
        result = pd.DataFrame({'Id':test_data['Id'].as_matrix(), 'SalePrice':predictions})
        result.to_csv(self.dir_name + "/" + csv_name, index=False)


if __name__ == "__main__":
    dir_name = os.path.dirname(os.path.realpath(__file__))
    obj = Solution(dir_name, dir_name + '/train.csv', dir_name + '/test.csv')
    obj.load_data()
    #obj.run()
    #obj.xgb_regressor()
    #obj.xgb_grid_search()
    obj.lightgbm_gird_search()