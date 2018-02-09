import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

def load_data(filename, skiprows = 1):
    """
    Function loads data stored in the file filename and returns it as a numpy ndarray.

    Inputs:
        filename: given as a string.

    Outputs:
        Data contained in the file, returned as a numpy ndarray
    """
    return np.loadtxt(filename, skiprows=skiprows, delimiter=' ')

def get_score(clf, X_train, Y_train):
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)
    ts = 0
    vs = 0
    train_score = 0
    val_score = 0
    for split in kf.split(X_train, Y_train):
        train, val = split
        clf.fit(X_train[train], Y_train[train])
        ts = clf.score(X_train[train], Y_train[train])
        vs = clf.score(X_train[val], Y_train[val])
        print('Train score: %f' % ts)
        print('Val score: %f' % vs)
        train_score += ts/n_splits
        val_score += vs/n_splits
    return train_score, val_score

def length_add(X):
    row_counts = np.sum(X, axis=1).reshape(-1, 1)
    return X*np.sqrt(row_counts)
    

data = load_data('data/training_data.txt');
print('Loaded training data!')

X_train = data[:, 1:]
Y_train = data[:,0]
Y_train = Y_train*2 - 1 # To get from -1 to 1
rand_ind = np.arange(len(X_train))
np.random.shuffle(rand_ind)
X_train = X_train[rand_ind]
Y_train = Y_train[rand_ind]

#base_estimator = DecisionTreeClassifier(max_depth=2)
clf = AdaBoostClassifier(n_estimators=400)
get_score(clf, X_train, Y_train)
