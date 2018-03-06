__author__ = 'tanxing01'
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils import check_array
import numpy as np
import pandas as pd
from scipy import sparse


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
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values



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
        test_ratio = 0.2
        if use_level is True:
             # 分层抽样, 构建基准特征
            feature_name = 'income_cat'
            self.data[feature_name] = np.ceil(self.data['media_income'] / 1.5)
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

    def run(self):
        self.split_train_test(use_level=True)

        y_feature_name = 'median_house_value'
        X = self.train_set.drop(y_feature_name, axis=1)
        x_labels = self.train_set[y_feature_name].copy()

        attr_addr = CombinedAttributesAdder(add_bedrooms_per_room=False)
        housing_extra_attribs = attr_addr.transform(self.data.values)
        housing_extra_attribs = pd.DataFrame(housing_extra_attribs, columns=list(self.data.columns) + ["rooms_per_household", "population_per_household"])

        cat_feature_name = 'ocean_proximity'
        housing_num = self.train_set.drop(cat_feature_name, axis=1)

        num_attribs = list(housing_num)
        cat_attribs = [cat_feature_name]

        num_pipeline = Pipeline([
            ('selector', DataFrameSelector(num_attribs)),
            ('imputer', Imputer(strategy='median')),
            ('attribs_addr', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

        cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(cat_attribs)),
            ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
        ])

        full_pipeline = FeatureUnion(transformer_list=[
            ('num_pipeline', num_pipeline),
            ('cat_pipeline', cat_pipeline),
        ])
        housing_prepared = full_pipeline.fit_transform(self.train_set)







