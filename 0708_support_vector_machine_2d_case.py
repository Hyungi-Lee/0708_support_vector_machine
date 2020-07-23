# https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html#sphx-glr-auto-examples-svm-plot
# -separating-hyperplane-py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
# 40개의 smaple을 만들어주는데
# 여기서 centers는 y의 값이 2개로 구분되어 있는 것을 말한다.
# 즉 cluster가 2개인 경우이다.
# random_state = 6은 6이라는 숫자를 넣어 여러 함수를 거쳐 random number를
# generate하기 위해 넣어주는 것이다. 
X, y = make_blobs(n_samples=40, centers=2, random_state=6)
clf = SVC(kernel='linear', C=1000)
clf.fit(X, y)

# 위에서 얻은 점들을 x0, x1 coordinate들로 나누어 주고 그것을
# plt에 젛어주는 과정
plt.scatter(X[:,0], X[:,1], c=y, s=30, cmap=plt.cm.Paired)
# axes를 잡아주기 즉 저 그림 하나를 잡아준다.
ax = plt.gca()

# x축의 왼쪽 끝 부터 오른쪽 끝까지의 범위
xlim = ax.get_xlim()
# y축의 왼쪽 끝 부터 오른쪽 끝까지의 범위
ylim = ax.get_ylim()

# xx는 xlim의 왼쪽 끝에서 오른쪽 끝까지의 범위를 일정한 간격으로 30개로 나누어 저장해
# 준 것이다.
xx = np.linspace(xlim[0], xlim[1], 30)
# yy는 ylim의 왼쪽 끝에서 오른쪽 끝까지의 범위를 일정한 간격으로 30개로 나누어 저장해
# 준 것이다.
yy = np.linspace(ylim[0], ylim[1], 30)
# np.meshgrid(yy,xx)는 만약 y가 [y1,y2,...,ym]이고 x가 [x1,...,xn] 이면
# [array([[y1,...,ym],[y1,...,ym],...,[y1,...,ym]]),             #n개
# array([[x1,...,x1],[x2,...,x2],...,[xn,...,xn]])]              #각각의 list에 m개의 xi가 들어 있음
YY, XX = np.meshgrid(yy, xx)
# 여기서 xy는 [x_i,y_j] for all i, for all j를 모두 포함한 list이다.
# 순서는 [x_1,y_1], [x_1, y_2],...,[x_1,y_m],[x_2,y_1],...,[x_n,y_m]의 순이다.
xy = np.vstack([XX.ravel(), YY.ravel()]).T

Z = clf.decision_function(xy).reshape(XX.shape)

# 그래프에 선을 그려주는 부분
# XX, YY, Z는 coordinate과 함수값을 나타내는 것이고 levels는 이 함수를 이용해서 그리고 싶은
# 함수 값의 선이다. 즉 -1은 f(x1,x2) = -1인 직선을 찾아서 계산하는 것이다.
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
        linestyles=['--', '-', '--'])

# suppot vector를 결정하는 점들에 o를 쳐주는 부분
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:,1], s=100,
        linewidth=1, facecolors='none', edgecolors='k')

plt.show()
