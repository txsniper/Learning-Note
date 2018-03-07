__author__ = 'tanxing01'
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils import check_array
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR



class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out


rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, population_ix] / X[:, household_ix]
            ret = np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            return ret
        else:
            ret = np.c_[X, rooms_per_household, population_per_household]
            return ret


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ret = X[self.attribute_names].values
        return ret



class Housing(object):
    def __init__(self, data_path):
        self.csv_data_path = data_path
        self.data = None
        self.train_set = None
        self.test_set  = None

    def load_data(self):
        self.data = pd.read_csv(self.csv_data_path)

    def split_train_test_random(self, test_ratio):
        shuffled_indices = np.random.permutation(len(self.data))
        test_set_size = int(len(self.data) * test_ratio)
        self.test_set = shuffled_indices[:test_set_size]
        self.train_set = shuffled_indices[test_set_size:]

    def split_train_test_level(self, feature_name, test_ratio):
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
        for train_index, test_index in split.split(self.data, self.data[feature_name]):
            self.train_set = self.data.loc[train_index]
            self.test_set  = self.data.loc[test_index]

    # split_method : 抽样方法 ;
    # feature_name : 如果是分层抽样，表示基准特征名;
    # test_ratio : 测试集占比
    #
    def split_train_test_internal(self, split_method, feature_name, test_ratio):
        if self.data is None:
            self.load_data()
        if split_method == 'random':
            self.split_train_test_random(test_ratio)
        elif split_method == 'level':
            self.split_train_test_level(feature_name, test_ratio)

    def split_train_test(self, use_level):
        if self.data is None:
            self.load_data()
        test_ratio = 0.2
        if use_level is True:
             # 分层抽样, 构建基准特征
            feature_name = 'income_cat'
            self.data[feature_name] = np.ceil(self.data['median_income'] / 1.5)
            self.data[feature_name].where(self.data[feature_name] < 5, 5.0, inplace=True)
            self.split_train_test_internal('level', feature_name, test_ratio)
            for set_ in (self.test_set, self.train_set):
                set_.drop(feature_name, axis=1, inplace=True)
        else:
            self.split_train_test_internal('random', None, test_ratio)

    def add_feature(self):
        self.data['rooms_per_household'] = self.data['total_rooms'] / self.data['households']
        self.data['bedrooms_per_room'] = self.data['total_bedrooms'] / self.data['total_rooms']
        self.data['population_per_household'] = self.data['population'] / self.data['households']


    def handle_features(self, set_):

        # step1: 处理缺失值
        #median = set_['total_bedrooms'].median()
        #set_['total_bedrooms'].fillna(median, inplace=True)
        imputer = Imputer(strategy='median')

        # remove categroy feature
        cat_feature_name = 'ocean_proximity'
        set_num = set_.drop(cat_feature_name, axis=1)
        imputer.fit(set_num)
        X = imputer.transform(set_num)
        set_tr = pd.DataFrame(X, columns=set_num.columns)


        # step2: 处理类别值
        set_cat = set_[cat_feature_name]
        #set_cat_encoded, set_categories = set_cat.factorize()
        #encoder = OneHotEncoder()
        #set_cat_1hot = encoder.fit_transform(set_cat_encoded.reshape(-1, 1))

        cat_encoder = CategoricalEncoder()
        set_cat_reshaped = set_cat.values.reshape(-1, 1)
        set_cat_1hot = cat_encoder.fit_transform(set_cat_reshaped)

    def display_error(self, scores, name):
            print(name + "Scores:", scores)
            print(name + "Mean:", scores.mean())
            print(name + "Standard deviation:", scores.std())

    # 线性模型
    def lin_reg_prediction(self, housing_prepared, housing_labels):
        lin_reg = LinearRegression()
        lin_reg.fit(housing_prepared, housing_labels)
        housing_predictions = lin_reg.predict(housing_prepared)
        # 平方误差
        lin_mse = mean_squared_error(housing_labels, housing_predictions)
        lin_rmse = np.sqrt(lin_mse)
        print("lin_reg rmse: " + str(lin_rmse))
        # 绝对误差
        lin_mae = mean_absolute_error(housing_labels, housing_predictions)
        print("lin_reg mae: " + str(lin_mae))

        # 10折交叉验证，使用平方误差，这里得出的是负值，下面使用 -lin_scores
        lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
            scoring="neg_mean_squared_error", cv=10)
        lin_rmse_scores = np.sqrt(-lin_scores)
        self.display_error(lin_rmse_scores, "lin_reg")

    # 决策树
    def tree_reg_prediction(self, housing_prepared, housing_labels):
        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(housing_prepared, housing_labels)
        housing_predictions = tree_reg.predict(housing_prepared)
        tree_mse = mean_squared_error(housing_labels, housing_predictions)
        tree_rmse = np.sqrt(tree_mse)
        print("tree_reg rmse: " + str(tree_rmse))
        scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
            scoring="neg_mean_squared_error", cv=10)
        tree_rmse_scores = np.sqrt(-scores)
        self.display_error(tree_rmse_scores, "tree_reg")

    def random_search_tree_reg(self, housing_prepared, housing_labels, attributes):
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import randint

        param_distribs = {
            'max_depth': randint(low=1, high=20),
            'max_features': randint(low=1, high=12),
        }
        tree_reg = DecisionTreeRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(tree_reg, param_distributions=param_distribs,
                        n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
        rnd_search.fit(housing_prepared, housing_labels)
        feature_importances = rnd_search.best_estimator_.feature_importances_
        print("feature_importances-------------------------")
        print(feature_importances)
        ret = sorted(zip(feature_importances, attributes), reverse=True)
        print(ret)
        return rnd_search.best_estimator_

    # 随机森林
    def forest_reg_prediction(self, housing_prepared, housing_labels):
        forest_reg = RandomForestRegressor(random_state=42)
        forest_reg.fit(housing_prepared, housing_labels)
        housing_predictions = forest_reg.predict(housing_prepared)
        forest_mse = mean_squared_error(housing_labels, housing_predictions)
        forest_rmse = np.sqrt(forest_mse)
        print("forest_rmse: "+ str(forest_rmse))
        forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
            scoring="neg_mean_squared_error", cv=10)
        forest_rmse_scores = np.sqrt(-forest_scores)
        self.display_error(forest_rmse_scores, "forest_rmse_scores")

    # svm 回归
    def svr_reg_prediction(self, housing_prepared, housing_labels):
        svm_reg = SVR(kernel="linear")
        svm_reg.fit(housing_prepared, housing_labels)
        housing_predictions = svm_reg.predict(housing_prepared)
        svm_mse = mean_squared_error(housing_labels, housing_predictions)
        svm_rmse = np.sqrt(svm_mse)
        print("svm_rmse: " + str(svm_rmse))

    def grid_search_svr_reg(self, housing_prepared, housing_labels, attributes):
        from sklearn.model_selection import GridSearchCV
        param_grid = [
            {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
            {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
                'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
        ]
        svm_reg = SVR()
        grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=4)
        grid_search.fit(housing_prepared, housing_labels)
        print(grid_search.best_estimator_)
        feature_importances = grid_search.best_estimator_.feature_importances_
        print("feature_importances-------------------------")
        print(feature_importances)
        ret = sorted(zip(feature_importances, attributes), reverse=True)
        print(ret)
        return grid_search.best_estimator_


    # 使用网格搜索来寻找最佳超参数
    def grid_search_forest_reg(self, housing_prepared, housing_labels, attributes):
        from sklearn.model_selection import GridSearchCV
        param_grid = [
            # try 12 (3×4) combinations of hyperparameters
            {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
            # then try 6 (2×3) combinations with bootstrap set as False
            {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
        ]
        forest_reg = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
            scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(housing_prepared, housing_labels)
        print(grid_search.best_estimator_)
        #print(grid_search.best_params_)
        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)

        feature_importances = grid_search.best_estimator_.feature_importances_
        print("feature_importances-------------------------")
        print(feature_importances)
        ret = sorted(zip(feature_importances, attributes), reverse=True)
        print(ret)
        return grid_search.best_estimator_

    # 随机搜索寻找最佳超参数
    def random_search_forest_reg(self, housing_prepared, housing_labels, attributes):
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import randint

        param_distribs = {
            'n_estimators': randint(low=1, high=200),
            'max_features': randint(low=1, high=8),
        }
        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                        n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
        rnd_search.fit(housing_prepared, housing_labels)
        cvres = rnd_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)

        feature_importances = rnd_search.best_estimator_.feature_importances_
        print("feature_importances-------------------------")
        print(feature_importances)
        ret = sorted(zip(feature_importances, attributes), reverse=True)
        print(ret)
        return rnd_search.best_estimator_

    def run(self):
        # step1: 构建训练集与测试集
        self.split_train_test(use_level=True)

        # step2: 拆分feature和label
        y_feature_name = 'median_house_value'
        X = self.train_set.drop(y_feature_name, axis=1)
        x_labels = self.train_set[y_feature_name].copy()

        '''
        attr_addr = CombinedAttributesAdder(add_bedrooms_per_room=False)
        housing_extra_attribs = attr_addr.transform(self.data.values)
        housing_extra_attribs = pd.DataFrame(housing_extra_attribs, columns=list(self.data.columns) + ["rooms_per_household", "population_per_household"])
        '''
        # step3: 区分数值特征和类别特征，对类别特征使用one-hot编码
        cat_feature_name = 'ocean_proximity'
        housing_num = X.drop(cat_feature_name, axis=1)

        num_attribs = list(housing_num)
        print(num_attribs)
        cat_attribs = [cat_feature_name]

        # step4:    num_pipeline, 数值类型特征处理流水线
        #           DataFrameSelector : 特征选择器，选择数值类型特征
        #           Imputer : 使用中间值填充缺失的特征值
        #           CombinedAttributesAdder : 增加联合特征，由原始特征计算得出
        #           StandardScaler : 特征归一化
        num_pipeline = Pipeline([
            ('selector', DataFrameSelector(num_attribs)),
            ('imputer', Imputer(strategy='median')),
            ('attribs_addr', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

        # CategoricalEncoder : 类别特征做onhot编码
        cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(cat_attribs)),
            ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
        ])
        # FeatureUnion中的对象可以并行操作
        full_pipeline = FeatureUnion(transformer_list=[
            ('num_pipeline', num_pipeline),
            ('cat_pipeline', cat_pipeline),
        ])

        # 数据处理
        housing_prepared = full_pipeline.fit_transform(X)

        # step5 : 多种模型预测，使用默认超参数
        self.lin_reg_prediction(housing_prepared, x_labels)
        self.tree_reg_prediction(housing_prepared, x_labels)
        self.forest_reg_prediction(housing_prepared, x_labels)
    
        '''
        scores = cross_val_score(lin_reg, housing_prepared, x_labels, scoring="neg_mean_squared_error", cv=10)
        pd.Series(np.sqrt(-scores)).describe()
        '''

        # step6: 使用网格搜索和随机搜索来寻找最佳超参数
        extra_attribs = ["rooms_per_household", "population_per_household", "bedrooms_per_room"]
        cat_encoder = cat_pipeline.named_steps["cat_encoder"]
        cat_one_hot_attribs = list(cat_encoder.categories_[0])
        attributes = num_attribs + extra_attribs + cat_one_hot_attribs
        #best_model = self.grid_search_forest_reg(housing_prepared, x_labels, attributes)
        #best_model = self.random_search_forest_reg(housing_prepared, x_labels, attributes)
        best_mddel = self.grid_search_svr_reg(housing_prepared, x_labels, attributes)

        # step7: 将模型应用到测试集
        X_test = self.test_set.drop(y_feature_name, axis=1)
        y_test = self.test_set[y_feature_name].copy()

        X_test_prepared = full_pipeline.transform(X_test)
        X_test_prediction = best_model.predict(X_test_prepared)
        final_mse = mean_squared_error(y_test, X_test_prediction)
        final_rmse = np.sqrt(final_mse)
        print("final_mse  : " + str(final_mse))
        print("final_rmse : " + str(final_rmse))
        
if __name__ == "__main__":
    csv_data_path = "./datasets/housing/housing.csv"
    housing_obj = Housing(csv_data_path)
    housing_obj.run()
