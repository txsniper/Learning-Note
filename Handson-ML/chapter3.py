from sklearn.datasets import fetch_mldata
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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

    def plot_predision_recall_vs_threshold(self, precisions, recalls, thresholds):
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
        plt.xlabel("Threshold")
        plt.legend(loc="upper left")
        plt.ylim([0, 1])

    def plot_roc_curve(self, fpr, tpr, label=None):
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()

    def classfier_5(self):
        y_train_5 = (self.y_train == 5)
        y_test_5  = (self.y_test  == 5)

        from sklearn.linear_model import SGDClassifier
        sgd_clf = SGDClassifier(max_iter=10, random_state=42)
        sgd_clf.fit(self.X_train, y_train_5)

        #self.my_cross_val_score(sgd_clf, y_train_5)

        #ret = cross_val_score(sgd_clf, self.X_train, y_train_5, cv=10, scoring="accuracy")
        #print(ret)
        y_train_pred = cross_val_predict(sgd_clf, self.X_train, y_train_5, cv=10)
        print(confusion_matrix(y_train_5, y_train_pred))
        print(precision_score(y_train_5, y_train_pred))
        print(recall_score(y_train_5, y_train_pred))
        print(f1_score(y_train_5, y_train_pred))

        y_scores = cross_val_predict(sgd_clf, self.X_train, y_train_5, cv=10, method="decision_function")
        print(y_scores.shape)
        precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
        self.plot_predision_recall_vs_threshold(precisions, recalls, thresholds)
        plt.xlim([-700000, 700000])
        plt.show()

        fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
        self.plot_roc_curve(fpr, tpr)

        from sklearn.ensemble import RandomForestClassifier
        forest_clf = RandomForestClassifier(random_state=42)
        y_probas_forest = cross_val_predict(forest_clf, self.X_train, y_train_5, cv=10, method="predict_proba")
        y_scores_forest = y_probas_forest[:, 1]
        fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
        plt.plot(fpr, tpr, "b:", label="SGD")
        self.plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
        plt.legend(loc="bottom right")
        plt.show()


if __name__ == "__main__":
    obj = Mnist()
    obj.split_train_test()
    obj.classfier_5()

