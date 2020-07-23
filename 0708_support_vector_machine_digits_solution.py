import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.model_selection import KFold
from itertools import product

# 먼저 data들을 전부 X와 y에 옮겨 담아 준다.
X, y = load_digits(return_X_y=True)
#y[y<5] = -1
#y[y>5] = 1

# 전체 sample의 길이를 n_sample에 저장해 준다.
n_sample = len(X)
# 그 중 80퍼센트를 training data로 이용해 주기 위해 n_train변수에 저장해 준다.
n_train = int(0.8*n_sample)
# X_train에 X의 앞에서 부터 n_train개 만큼 slice해서 저장해 준다.
X_train = X[:n_train]
# y_train에 y의 앞에서 부터 n_train개 만큼 slice해서 저장해 준다.
y_train = y[:n_train]
# X_test에 나머지 X를 전부 저장해 준다.
X_test = X[n_train:]
# y_test에 나머지 y를 전부 저장해 준다.
y_test = y[n_train:]
# range of hyperparameters
# hyperparameter들의 조율을 np.logspace를 이용하여 완료하려 한다.
# np.logspace(x,y,n)은 x와 y사이를 일정한 n개의 간걱으로 나누어 n+1개의 숫자를 만들고
# 그것들을 지수로 하는 10의 승을 구한 결과값을 list로 저장한다.
gamma_range = np.logspace(-9, 3, 10)
C_range = np.logspace(-2, 4, 10)

# K-fold cross validation
# validation set을 이용하기 위해 training set을 cross validation을 이용하여
# 나누어 주려 한다. 그러기 위해서 import 해놓은 KFold를 이용해 준다.
# n_split개 만큼의 set으로 shuffle하여 나누겠다는 소리이다.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# fit SVM for all parameters
# 평균적으로 가장 높은 정확도를 보이는 것들을 저장하여 사용하기 위해
# 정확도를 저장할 bset_acc, gamma값을 저장할 best_gamma, C값을 저장할 best_C를 만들어 준다.
# 여기서 gamma는 gaussian kernel의 hyperparameter이고 C는 soft margin의 hyperparameter이다.
best_acc = 0.
best_gamma = 0.
best_C = 0.
# product을 이용하여 [gamma_i,C_j] for all i for all j에 대해 전부 for문을 돌아준다.
for gamma, C in product(gamma_range, C_range):
    # cross validation에 대해서 얻은 accuaracy의 평균을 계산해 주기 위해 변수를 준비한다.
    avg_acc = 0.
    # kf.split을 이용하여 cross validation의 값을 for문을 사용하며 이용해준다.
    # train_idx에는 training에 사용될 index들이 val_idx에는 validation에 사용될 index들이
    # 저장되어 있다.
    for train_idx, val_idx in kf.split(X_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        # svm을 이용하기 위해 clf를 만들어 준다.
        # kernel은 gaussian kernel을 이용하고 gamma는 gamma, C는 C를 넣어주면 된다.
        clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)
        # clf를 training시킨다.
        clf.fit(X_tr, y_tr)
        # training된 clf를 이용하여 validation set에서의 accuaracy를 avg_acc에
        # 더해준다.
        avg_acc += clf.score(X_val, y_val)
    # 이렇게 얻어진 av_acc를 5를 나누어 주어 평균을 계산한다.
    avg_acc /= 5.
    # 그리고 이 때의 gamma, C그리고 avg_acc를 print해준다.
    print('gamma %.4f C %.4f acc %.4f' % (gamma, C, avg_acc))
    # 그리고 평균 accuaracy가 best_acc보다 크면 gamma, C, accuaracy를 모두 update해준다.
    if avg_acc > best_acc:
        best_acc = avg_acc
        best_gamma = gamma
        best_C = C

print('best gamma: %.4f' % (best_gamma))
print('best C: %.4f' % (best_C))

clf = svm.SVC(kernel='rbf', gamma=best_gamma, C=best_C)
clf.fit(X_train, y_train)
print('test acc: %.4f' % (clf.score(X_test, y_test)))
