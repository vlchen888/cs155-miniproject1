import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold

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

def print_output(Y_test):
    f = open('data/svm.txt', 'w')
    f.write('Id,Prediction\n')
    for i in range(len(Y_test)):
        f.write(str(i+1)+','+str(int(Y_test[i])) + '\n')
    f.close()

def run_test(clf, tf_transformer):
    data = load_data('data/test_data.txt')
    print('Loaded test data!')
    X_test_tf = tf_transformer.transform(data)
    Y_test = clf.predict(X_test_tf)
    print_output(Y_test)

data = load_data('data/training_data.txt');
print('Loaded training data!')

X_train = data[:, 1:]
Y_train = data[:,0]

tf_transformer = TfidfTransformer().fit(X_train)
X_train_tf = tf_transformer.transform(X_train)
clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, max_iter=400, tol=None)
clf.fit(X_train_tf, Y_train)
run_test(clf, tf_transformer)
