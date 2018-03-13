import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.svm import LinearSVC
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVC

class SVM(object):
    def __init__(self):
        pass
    
    def load_data(self):
        self.X, self.y = make_moons(n_samples=100, noise=0.15, random_state=42)

    def svc_soft_margin(self):
        data_set = datasets.load_iris()
        X = data_set['data'][:, (2,3)]
        y = (data_set['target'] == 2).astype(np.float64)
        svm_clf = Pipeline([
            ('scaler', StandardScaler()),
            ('linear_svc', LinearSVC(C=1, loss='hinge')),
        ])
        svm_clf.fit(X, y)
        ret = svm_clf.predict([[5.5, 1.7]])
        print(ret)

    def plot_dataset(X, y, axes):
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
        plt.axis(axes)
        plt.grid(True, which='both')
        plt.xlabel(r"$x_1$", fontsize=20)
        plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

    def plot_predictions(X, y, clf, axes):
        x0s = np.linspace(axes[0], axes[1], 100)
        x1s = np.linspace(axes[2], axes[3], 100)
        x0, x1 = np.meshgrid(x0s, x1s)
        X = np.c_[x0.ravel(), x1.ravel()]
        y_pred = clf.predict(X).reshape(x0.shape)
        y_decision = clf.decision_function(X).reshape(x0.shape)
        plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
        plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

    def svm_rbf_kernel(self):
        rbf_kernel_svm_clf = Pipeline((
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel='rbf', gamma=5, C=0.001))
        ))
        rbf_kernel_svm_clf.fit(X, y)
        gamma1, gamma2 = 0.1, 5
        hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)
        svm_clfs = []
        for gamma, C in hyperparams:
            rbf_kernel_svm_clf = Pipeline([
                ('scaler', StandardScaler()),
                ('svm_clf', SVC(kernel='rbf', gamma=gamma=, C=C))
            ])
            rbf_kernel_svm_clf.fit(X, y)
            svm_clfs.append(rbf_kernel_svm_clf)
        plt.figure(figsize=(11, 7))
        for i, svm_clf in enumerate(svm_clfs):
            plt.subplot(221 + i)
            self.plot_predictions(svm_clf, [-1.5, 2.5, -1, 1.5])
            self.plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
            gamma, C = hyperparams[i]
            plt.title(r"$\gamma={}, C={}$".format(gamma, C), fontsize=16)
        plt.show()
            

    def svm_polynomial(self):
        from sklearn.preprocessing import PolynomialFeatures
        X = self.X
        y = self.y
        
        plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
        #plt.show()
        
        polynomial_svm_clf = Pipeline([
            ('poly_features', PolynomialFeatures(degree=3)),
            ('scaler', StandardScaler()),
            ('svm_clf', LinearSVC(C=10, loss='hinge', random_state=42))
        ])
        polynomial_svm_clf.fit(X, y)
        

        self.plot_predictions(X, y, polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
        plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
        plt.show()

        poly_kernel_svm_clf = Pipeline((
            ('scaler', StandardScaler()),
            ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5))
        ))
        poly_kernel_svm_clf.fit(X, y)
        poly100_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="poly", degree=10, coef0=100, C=5))
        ])
        poly100_kernel_svm_clf.fit(X, y)
        plt.figure(figsize=(11, 4))
        plt.subplot(121)
        plot_predictions(poly_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
        plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
        plt.title(r"$d=3, r=1, C=5$", fontsize=18)

        plt.subplot(122)
        plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
        plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
        plt.title(r"$d=10, r=100, C=5$", fontsize=18)
        plt.show()



if __name__ == '__main__':
    obj = SVM()
    obj.load_data()
    #obj.svc_soft_margin()
    #obj.svm_polynomial()
    obj.svm_rbf_kernel()