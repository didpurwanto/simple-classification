import cv2
import numpy as np
import os
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn import metrics



bibnumbers = []
with open("test.txt", "r") as f:
	for line in f:
		bibnumbers.append(line.strip())


src_train = np.zeros(1)
src2_train = np.zeros(2)
des_train = np.zeros(1)
des2_train = np.zeros(2)
pro_train = np.zeros(2)
lth_train = np.zeros(1)
cll_train = np.zeros(1)

src_test = np.zeros(1)
src2_test = np.zeros(2)
des_test = np.zeros(1)
des2_test = np.zeros(2)
pro_test = np.zeros(2)
lth_test = np.zeros(1)
cll_test = np.zeros(1)
count = 0
for data in bibnumbers:
	tmp = data.split('	')
	src_port = np.array([tmp[0]])
	src2 = np.array([tmp[1]])
	des_port = np.array([tmp[2]])
	des2 = np.array([tmp[3]])
	protocol = np.array([tmp[4]])
	length = np.array([tmp[5]])
	outclass = np.array([tmp[7]])

	# one hot encoding

	if protocol == "TCP":
		protocol = np.array([0,1])
	else:
		protocol = np.array([1,0])

	if src2 == "Wellknown":
		src2 = np.array([1,0])
	else:
		src2 = np.array([0,1])

	if des2 == "Wellknown":
		des2 = np.array([1,0])
	else:
		des2 = np.array([0,1])

	if outclass == 'NGBR_Expedited':
		outclass = np.array([1])
	elif outclass == 'NGBR_Assured':
		outclass = np.array([2])
	else:
		outclass = np.array([3])
	
	count+= 1
	

	if count<21:
		src_train = np.vstack((src_train, src_port))
		src2_train = np.vstack((src2_train, src2))
		des_train = np.vstack((des_train, des_port))
		des2_train = np.vstack((des2_train, des2))
		pro_train = np.vstack((pro_train, protocol))
		lth_train = np.vstack((lth_train, length))
		cll_train = np.vstack((cll_train, outclass))
	else:
		src_test = np.vstack((src_test, src_port))
		src2_test = np.vstack((src2_test, src2))
		des_test = np.vstack((des_test, des_port))
		des2_test = np.vstack((des2_test, des2))
		pro_test = np.vstack((pro_test, protocol))
		lth_test = np.vstack((lth_test, length))
		cll_test = np.vstack((cll_test, outclass))


X_train = np.hstack((des_train, pro_train, src2_train, lth_train))
X_test = np.hstack(( des_test, pro_test, src2_test, lth_test))
Y_train = cll_train
Y_test = cll_test

# print(X_train)
# print(Y_train)


lin_clf = svm.LinearSVC()
lin_clf.fit(X_train, Y_train)


#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, Y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print(y_pred)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
