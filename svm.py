import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tabulate import tabulate
import sys

if(len(sys.argv)>1):
	if(len(sys.argv)==2):
		default=True
		try:
			tryfloat=float(sys.argv[1])
			if(tryfloat>=1):
				print("Data test too big, automatically resized into 0.9")
				tryfloat=0.9
		except Exception:
			tryfloat=0.33
			default=False

		if(default):
			from sklearn import datasets
			iris = datasets.load_iris()
			x = iris.data
			y = iris.target

			feature_names = iris.feature_names
			target_names = iris.target_names
		else:
			x = pd.read_csv(str(sys.argv[1]))
			xnp = np.array(x)

			feature_names=[y for y in x.keys()]
			target_names=['no','yes']
			
			x = xnp[:,:-1] #data
			y = xnp[:,-1] #class filled with 0/1

		test_size=tryfloat
	elif(len(sys.argv)==3):
		try:
			tryfloat=float(sys.argv[2])
			if(tryfloat>=1):
				print("Data test too big, automatically resized into 0.9")
				tryfloat=0.9
		except Exception:
			tryfloat=0.33

		x = pd.read_csv(str(sys.argv[1]))
		xnp = np.array(x)

		feature_names=[y for y in x.keys()]
		target_names=['no','yes']
		
		x = xnp[:,:-1] #data
		y = xnp[:,-1] #class filled with 0/1
		test_size=tryfloat
else:
	from sklearn import datasets
	iris = datasets.load_iris()

	x = iris.data
	y = iris.target

	feature_names = iris.feature_names
	target_names = iris.target_names

	test_size = 0.33

x_train, x_test, y_train , y_test = train_test_split(x,y,test_size=test_size , random_state=1)

svm = SVC(kernel='linear')

#creating model
svm.fit(x_train,y_train)
print("Training ", str(len(x_train))," datas...")

#generating prediction result
y_predic = svm.predict(x_test)
print("Testing ", str(len(x_test))," datas...")

#comparing prediction with the real datas which is resulting precentage of the accuracy
accuracy = metrics.accuracy_score(y_test,y_predic)

print("SVM Accuracy on",str(test_size*100)+"%","of","HA" ,"dataset is " + str(accuracy*100)+"%")

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