import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

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
    f = open('predictions/C1svm.csv', 'w')
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

ind = np.arange(len(data))
# Shuffle data
np.random.shuffle(ind)
X_train = data[ind, 1:]
Y_train = data[ind,0]

tf_transformer = TfidfTransformer().fit(X_train)
X_train_tf = tf_transformer.transform(X_train)

clf = SVC(C=1, gamma=1, kernel='rbf', max_iter=20000)
clf.fit(X_train_tf, Y_train)
run_test(clf, tf_transformer)

#Cs = [.01, .1, 1]
#gammas = [1]
#kernel = ['rbf']
#degrees = [2]
#max_iter = [10000]
#clf = SVC()
#gs = GridSearchCV(estimator=clf, param_grid=dict(C=Cs, gamma=gammas, kernel=kernel, degree=degrees, max_iter=max_iter), verbose=True)
#gs.fit(X_train_tf, Y_train)
#print(gs.best_params_)
#print(gs.best_score_)
#print(gs.cv_results_)



