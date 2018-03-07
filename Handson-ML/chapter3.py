from sklearn.datasets import fetch_mldata
import numpy as np

class Mnist(object):
    def __init__(self):
        pass

    def load_data(self):
        self.mnist = fetch_mldata('MNIST original')
        self.X = self.mnist['data']
        self.y = self.mnist['target']
    
    def split_train_test(self):
        self.load_data()
        self.X_train = self.X[:60000]
        self.X_test  = self.X[60000:]
        self.y_train = self.y[:60000]
        self.y_test  = self.y[60000:]
        shuffle_index = np.random.permutation(60000)
        self.X_train, self.y_train = self.X_train[shuffle_index], self.y_train[shuffle_index]


    def my_cross_val_score(self, model, y_train):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.base import clone
        skflods = StratifiedKFold(n_splits=3, random_state=42)
        for train_index, test_index in skflods.split(self.X_train, y_train):
            clone_clf = clone(model)
            X_train_flods = self.X_train[train_index]
            y_train_flods = (y_train[train_index])
            X_test_flods = self.X_train[test_index]
            y_test_flods = (y_train[test_index])
            clone_clf.fit(X_train_flods, y_train_flods)
            y_test_pred = clone_clf.predict(X_test_flods)
            n_correct = sum(y_test_pred == y_test_flods)
            print(n_correct / len(y_test_pred))

    def classfier_5(self):
        y_train_5 = (self.y_train == 5)
        y_test_5  = (self.y_test  == 5)

        from sklearn.linear_model import SGDClassifier
        sgd_clf = SGDClassifier(random_state=42)
        sgd_clf.fit(self.X_train, y_train_5)

        self.my_cross_val_score(sgd_clf, y_train_5)



if __name__ == "__main__":
    obj = Mnist()
    obj.split_train_test()
    obj.classfier_5()

