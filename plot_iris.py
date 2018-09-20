"""
==================================
EDITED CODE FROM SCI-KIT LEARN*
==================================

Hi, this is an edited code from
http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
to show how SVM** group iris dataset
based on its own feature name!

PRE-REQUISITES : Please install these libraries.
sci-kit learn, tabulate, numpy, matplotlib 

*Genuine code on the link above
**SVC Linear Kernel
***Default train size is 0.67 , but you can set it using -size='value'

Edited by @akbarnotopb 
19/9/2018
V.02
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tabulate import tabulate
import sys
import time


#These code belongs to scikit-learn documentation
def make_meshgrid(x, y, h=.02):
  x_min, x_max = x.min() - 1, x.max() + 1
  y_min, y_max = y.min() - 1, y.max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))
  return xx, yy

#These code belongs to scikit-learn documentation
def plot_contours(ax, clf, xx, yy, **params):
  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  out = ax.contourf(xx, yy, Z, **params)
  return out

if(len(sys.argv)==2 and sys.argv[1][:6]=="-size="):
  print("Hoyak!")
  test_size=1-float(sys.argv[1][6:])
else:
  test_size=0.33

# test_size=0.33
# import some data to play with
iris = datasets.load_iris()
y = iris.target


C = 1.0  # SVM regularization parameter
clf = svm.SVC(kernel='linear', C=C)


# Set-up 2x3 grid for plotting.
fig, sub = plt.subplots(2, 3)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

label=[['Sepal Length','Sepal Width'],['Sepal Length','Petal Length'],['Sepal Length','Petal Width'],
    ['Sepal Width','Petal Length'],['Sepal Width','Petal Width'],['Petal Length','Petal Width']]

idx=0
for ax in sub.flatten():
  if idx == 0:
    X=iris.data[: , :2]
  elif idx == 1:
    X=iris.data[:, [0,2]]
  elif idx == 2:
    X=iris.data[:, [0,3]]
  elif idx == 3:
    X=iris.data[:, [1,2]]
  elif idx==4:
    X=iris.data[:, [1,3]]
  elif idx ==5:
    X=iris.data[:, [2,3]]
  X0, X1 = X[:, 0], X[:, 1]

  x_train, x_test, y_train , y_test = train_test_split(X,y,test_size=test_size , random_state=1)
  clf.fit(x_train,y_train)
  y_predic=clf.predict(x_test)
  print(label[idx][0],"X",label[idx][1],"accuracy :", metrics.accuracy_score(y_test,y_predic))

  clf.fit(X,y)
  xx, yy = make_meshgrid(X0, X1)
  plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
  ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
  ax.set_xlim(xx.min(), xx.max())
  ax.set_ylim(yy.min(), yy.max())
  ax.set_xlabel(label[idx][0])
  ax.set_ylabel(label[idx][1])
  ax.set_xticks(())
  ax.set_yticks(())
  idx+=1

x = iris.data
y = iris.target

#splitting the dataset
x_train, x_test, y_train , y_test = train_test_split(x,y,test_size=test_size , random_state=1)

svk = SVC(kernel='linear')

print("\n~ Summary ~")

#creating model
print("Training ", str(len(x_train))," datas...")
start_time = time.time()
svk.fit(x_train,y_train)
end_time = time.time()-start_time
print("Training data took", str(end_time)[:6],"seconds.")


#generating prediction result
print("Testing ", str(len(x_test))," datas...")
start_time = time.time()
y_predic = svk.predict(x_test)
end_time = time.time() - start_time
print("Testing data took",str(end_time)[:6],"seconds.")


#comparing prediction with the real datas which is resulting precentage of the accuracy
accuracy = metrics.accuracy_score(y_test,y_predic)


print("SVM Accuracy on",str(test_size*100)[:4]+"%","of iris dataset is " + str(accuracy*100)[:4]+"%")

feature_names = iris.feature_names
target_names = iris.target_names

confmatrix = []
for i in range(0,len(target_names)):
  confmatrix.append([])
  if i==0:
    for j in range(0, len(target_names)):
      confmatrix[i].append(j)
    confmatrix.append([])
  for j in range(0,len(target_names)):
    if j==0:
      confmatrix[i+1].append(i)
    confmatrix[i+1].append(0)



for i in range(0, len(y_predic)):
  idx2 = y_predic[i]+1
  idx1 = y_test[i]+1
  if(idx1==idx2):
    confmatrix[idx1][idx2]+=1
  else:
    confmatrix[idx1][idx2]+=1

print(tabulate(confmatrix,headers="firstrow",tablefmt="grid"))

print("Details :")
for i in range(0, len(target_names)):
  print(str(i)+" "+target_names[i])

plt.show()