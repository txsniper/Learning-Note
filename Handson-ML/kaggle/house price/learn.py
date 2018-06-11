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
import lightgbm as lgb
import os

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
        data['HaveBsmt'] = 0
        data.loc[hBsmt_index,'HaveBsmt'] = 1
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

        #print(all_X.head())
        #return all_X
        num_train = self.train_data.shape[0]
        X_train = all_X[:num_train].as_matrix()
        X_test  = all_X[num_train:].as_matrix()
        y_train = self.train_data.SalePrice.as_matrix()
        return X_train, X_test, y_train

    
    def rmsle_cv(self, model, X, y):
        n_folds = 5
        kf = KFold(n_folds, shuffle=True, random_state=41).get_n_splits(X.values)
        rmse = np.sqrt(-cross_val_score(model, X.values, y, scoring='neg_mean_squared_error', cv=kf))
        return rmse


    def run(self):
        self.load_data()
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


if __name__ == "__main__":
    dir_name = os.path.dirname(os.path.realpath(__file__))
    obj = Solution(dir_name, dir_name + '/train.csv', dir_name + '/test.csv')
    obj.run()