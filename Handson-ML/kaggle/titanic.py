import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import roc_auc_score  
from sklearn.metrics import classification_report
from sklearn import model_selection
from scipy import sparse
import os

class Titanic(object):
    def __init__(self, curr_dir, csv_train, csv_test):
        self.curr_dir = curr_dir
        self.train_path =  csv_train
        self.test_path = csv_test

    def load_data(self):
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)

    def data_describe(self):
        self.train_data.info()
        ret = self.train_data.describe()
        print(ret)
        print(self.train_data.head())
        #self.train_data.hist(bins=50, figsize=(20, 15))
        #plt.show()

    def data_view_internal(self, name):
        print('-'*40)
        print(self.train_data.groupby([name,'Survived'])['Survived'].count())

    def data_view(self):
        feature_list = ['Sex', 'Pclass', 'Age']
        for feature_name in feature_list:
            self.data_view_internal(feature_name)

    
    def write_predictions_2_csv(self, test_data,  predictions, csv_name):
        result = pd.DataFrame({'PassengerId':test_data['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
        result.to_csv(self.curr_dir + "/" + csv_name, index=False)

    def get_one_hot_feature(self, data, name):
        return pd.get_dummies(data[name], prefix=name)

    def lr_with_validate(self, train_data): 
        split_train, split_cv = model_selection.train_test_split(train_data, test_size=0.3, random_state=42)
        
        split_train = split_train.drop(['PassengerId'], axis=1)
        X_train = split_train.values[:,1:]
        y_train = split_train.values[:,0]
        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        #
        X_test_temp = split_cv.loc[split_cv['PassengerId'] == 24]
        print(X_test_temp)
        X_test_temp = X_test_temp.drop(['PassengerId', 'Survived'], axis=1)
        X_test_predict = clf.predict(X_test_temp)
        print(X_test_predict)
        #
        X_test_all = split_cv.drop(['PassengerId'], axis=1)
        X_test = X_test_all.values[:,1:]
        y_test = X_test_all.values[:, 0]
        predictions = clf.predict(X_test)
        ret = split_cv[predictions != y_test]['PassengerId'].values
        origin_data = pd.read_csv(self.train_path)
        origin_test_data = origin_data.loc[origin_data['PassengerId'].isin(ret)]
        #print(origin_test_data)


    def preprocessing_feature(self, data):
        # age, Cabin, Fare
        # 增加Title 
        data['Title']=0
        data['Title']=data['Name'].str.extract('([A-Za-z]+)\.', expand=False) #lets extract the Salutations
        data['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                    ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'], inplace=True)

        # Age
        data.loc[(data.Age.isnull())&(data.Title=='Mr'),'Age']= data.Age[data.Title=="Mr"].mean()
        data.loc[(data.Age.isnull())&(data.Title=='Mrs'),'Age']= data.Age[data.Title=="Mrs"].mean()
        data.loc[(data.Age.isnull())&(data.Title=='Master'),'Age']= data.Age[data.Title=="Master"].mean()
        data.loc[(data.Age.isnull())&(data.Title=='Miss'),'Age']= data.Age[data.Title=="Miss"].mean()
        data.loc[(data.Age.isnull())&(data.Title=='Other'),'Age']= data.Age[data.Title=="Other"].mean()

        label_encoder = LabelEncoder()
        data['Age_Bin'] = pd.qcut(data['Age'], 5)
        data['Age_Bin'] = label_encoder.fit_transform(data['Age_Bin'])

        # Title
        data['Title'] = data['Title'].map( {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master':3, 'Other':4} )
        data['Title'] = data['Title'].fillna(data['Title'].mode().iloc[0])
        data['Title'] = data['Title'].astype(int)


        data.loc[data['Cabin'].notnull(), 'Cabin'] = 'YES'
        data.loc[data['Cabin'].isnull(), 'Cabin'] = 'No'

        data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
        data['Fare_Bin'] = pd.qcut(data['Fare'], 4)
        data['Fare_Bin'] = label_encoder.fit_transform(data['Fare_Bin'])

        # 类别特征one-hot编码
        #categorical_feature = ['Cabin', 'Embarked', 'Sex', 'Pclass']
        categorical_feature = ['Embarked', 'Sex', 'Pclass']
        one_hot_list = []
        for name in categorical_feature:
            feature = self.get_one_hot_feature(data, name)
            one_hot_list.append(feature)
        # 增加特征，数据清洗
        one_hot_features = pd.concat(one_hot_list, axis=1)
        data = pd.concat([data, one_hot_features], axis=1)
        data.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
        return data

    def random_forest(self):
        train_data = self.preprocessing_feature(self.train_data)
        #print(train_data.head())
        train_data_no_id = train_data.drop(['PassengerId'], axis=1)
        train_np = train_data_no_id.as_matrix()
        y = train_np[:, 0]
        X = train_np[:, 1:]
        param_grid = {
            'n_estimators' : [ 100, 130, 150, 180, 200, 300],
            'max_depth' : [3, 4, 5, 6, 7, 8],
            'max_leaf_nodes' : [2, 3, 4, 5],

        }
        scores = ['roc_auc']
        bclf = None
        for score in scores:
            rf = RandomForestClassifier(random_state=14)
            grid = GridSearchCV(rf, param_grid, cv=5, scoring='%s' % score, verbose=1)
            #grid = RandomizedSearchCV(rf, param_grid, cv=10, scoring='%s' % score, verbose=1)
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
        test_data = self.preprocessing_feature(self.test_data)
        test = test_data.drop(['PassengerId'], axis=1)
        predictions = bclf.predict(test.as_matrix())
        self.write_predictions_2_csv(test_data, predictions, "random_forest.csv")
        
    def grand_boost(self):
        train_data = self.preprocessing_feature(self.train_data)
        train_data_no_id = train_data.drop(['PassengerId'], axis=1)
        train_np = train_data_no_id.as_matrix()
        y = train_np[:, 0]
        X = train_np[:, 1:]
        gb = GradientBoostingClassifier(n_estimators=300, max_depth=10, learning_rate=0.01)
        gb.fit(X, y)
        cross_score = cross_val_score(gb, X, y, cv=5)
        print(cross_score)

        test_data = self.preprocessing_feature(self.test_data)
        test = test_data.drop(['PassengerId'], axis=1)
        #predictions = gb.predict(test.as_matrix())
        #self.write_predictions_2_csv(test_data, predictions, "grand_boost.csv")


    def lr(self, curr_dir):
        data = self.preprocessing_feature(self.train_data)
        train_data = data.filter(regex='PassengerId|Survived|Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
        

        # 训练集归一化
        scaler = StandardScaler()
        age_scale_param = scaler.fit(train_data['Age'].values.reshape(-1, 1))
        train_data['Age_scaled'] = scaler.fit_transform(train_data['Age'].values.reshape(-1, 1), age_scale_param)
        fare_scale_param = scaler.fit(train_data['Fare'].values.reshape(-1, 1))
        train_data['Fare_scaled'] = scaler.fit_transform(train_data['Fare'].values.reshape(-1, 1), fare_scale_param)
        train_data.drop(['Age', 'Fare'], axis=1, inplace=True)

        # 使用LR训练
        #print(train_data.head())
        train_data_no_id = train_data.drop(['PassengerId'], axis=1)
        train_np = train_data_no_id.as_matrix()
        print(train_np[0:5])
        #
        self.lr_with_validate(train_data)
        #
        y = train_np[:, 0]
        X = train_np[:, 1:]
        clf = LogisticRegression()
        clf.fit(X, y)
        #cross_score = cross_val_score(clf, X, y, cv=5)
        #print(cross_score)

        test_data = self.preprocessing_feature(self.test_data)
        test_data['Age_scaled'] = scaler.fit_transform(test_data['Age'].values.reshape(-1, 1), age_scale_param)
        test_data['Fare_scaled'] = scaler.fit_transform(test_data['Fare'].values.reshape(-1, 1), fare_scale_param)
        test_data.drop(['Age', 'Fare'], axis=1, inplace=True)
        test = test_data.filter(regex='Age*|SibSp|Parch|Fare*|Cabin*|Embarked*|Sex*|Pclass*')
        #print(test.head())

        '''
        self.write_predictions_2_csv(test_data, predictions, 'logistic_regression_predictions.csv')
        '''

if __name__ == "__main__":
    dir_name = os.path.dirname(os.path.realpath(__file__))
    obj = Titanic(dir_name, dir_name + '/train.csv', dir_name + '/test.csv')
    obj.load_data()
    obj.data_describe()
    #obj.lr(dir_name)
    #obj.random_forest()
    #obj.load_data()
    #obj.grand_boost()
    