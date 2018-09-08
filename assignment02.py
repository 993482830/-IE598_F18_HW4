#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 14:02:12 2018

@author: hurenjie
"""


from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print('Class labels:', np.unique(y))
#Class labels: [0 1 2]
   
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
   
print('Labels counts in y:', np.bincount(y))
#   Labels counts in y: [50 50 50]
print('Labels counts in y_train:', np.bincount(y_train))
#   Labels counts in y_train: [35 35 35]
print('Labels counts in y_test:', np.bincount(y_test))
#   Labels counts in y_test: [15 15 15]
   
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
   
y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
#   Misclassified samples: 3
   
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#   Accuracy: 0.93

print('Accuracy: %.2f' % ppn.score(X_test_std, y_test))
#   Accuracy: 0.93
   
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
def plot_decision_regions(X, y, classifier, test_idx=None,resolution=0.02):
       # setup marker generator and color map
       markers = ('s', 'x', 'o', '^', 'v')
       colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
       cmap = ListedColormap(colors[:len(np.unique(y))])
       # plot the decision surface
       x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
       x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
       xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                              np.arange(x2_min, x2_max, resolution))
       Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
       Z = Z.reshape(xx1.shape)
       plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
       plt.xlim(xx1.min(), xx1.max())
       plt.ylim(xx2.min(), xx2.max())
       for idx, cl in enumerate(np.unique(y)):
           plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                       alpha=0.8, c=colors[idx],
                       marker=markers[idx], label=cl,
                       edgecolor='black')
       # highlight test samples
       if test_idx:
           # plot all samples
           X_test, y_test = X[test_idx, :], y[test_idx]
           plt.scatter(X_test[:, 0], X_test[:, 1],
                       c='', edgecolor='black', alpha=1.0,
                       linewidth=1, marker='o',
                       s=100, label='test set')
           
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,
y=y_combined,
classifier=ppn,
test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
   
   
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(X_train, y_train)

#Decision Tree
#from sklearn import metrics
#y_pred = clf.predict(X_test)
import matplotlib.pyplot as plt
import numpy as np
def gini(p):
 return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))
def entropy(p):
 return - p*np.log2(p) - (1 - p)*np.log2((1 - p))
def error(p):
 return 1 - np.max([p, 1 - p])
x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                           ['Entropy','Entropy(scaled)','Gini Impunity','Misclassification error'],
                           ['-','-','--','-.'],
                           ['black','lightgray','red','green','cyan']):
    line=ax.plot(x,i,label=lab,linestyle=ls,lw=2,color=c)                           
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
           ncol=5, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='gini',
                                  max_depth=4,
                                  random_state=1)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined,
                          y_combined,
                          classifier=tree,
                          test_idx=range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()

#from pydotplus import graph_from_dot_data
#from sklearn.tree import export_graphviz
#dot_data = export_graphviz(tree,
#                           filled=True,
#                           rounded=True,
#                           class_name=['Setosa',
#                                       'Versicolor',
#                                       'Verginica',
#                                       ],
#                         feature_names=['petal length','petal width'],
#                         out_file=none)
#graph = graph_from_dot_data(dot_data)
#raph.write_png('tree.png')




#from sklearn.neighbors import KNeighborsClassifier
##try K=1 through K=25 and record testing accuracy
#k_range=range(1,26)
#scores=[]
#for k in k_range:
#    knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='minkowski')
#    knn.fit(X_train_std, y_train)
##    plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105,150))
#    print(scores)
##    plt.xlabel('petal length [standardized]')
##    plt.ylabel('petal width [standardized]')
##    plt.legend(loc='upper left')
##    plt.show()
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
#try K=1 through K=25 and record testing accuracy
k_range=range(1,26)
scores=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))
print(scores)
#    print(scores.append(metrics.accuracy_score(y_test,y_pred)))
      

print("My name is RENJIE HU")
print("My NetID is: 659740767")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")