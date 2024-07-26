# Alex Ivensky
# ECE 1395
# Homework 8

import numpy as np
from scipy.io import loadmat, savemat
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

data = loadmat("input/HW8_data1.mat")
X = data['X'] # 5000, 400
y = data['y'] # 5000, 1
labels = np.unique(y)

# random images
rand_pics = np.random.randint(0, 5000, size=25)
fig, ax = plt.subplots(5, 5, figsize=(10,10))
for i in range(5):
    for j in range(5):
        idx = 5*i + j
        img = X[rand_pics[idx], :].reshape(20, 20).T
        ax[i, j].set_title(y[rand_pics[idx], 0])
        ax[i, j].imshow(img, cmap='gray')
plt.savefig('output/ps8-1-a.png')

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(0.1))
combined_train = np.hstack((X_train, y_train))

# random bagging 
np.random.shuffle(combined_train)
X1 = X_train[:1000]
y_train1 = y_train[:1000]
np.random.shuffle(combined_train)
X2 = X_train[:1000]
y_train2 = y_train[:1000]
np.random.shuffle(combined_train)
X3 = X_train[:1000]
y_train3 = y_train[:1000]
np.random.shuffle(combined_train)
X4 = X_train[:1000]
y_train4 = y_train[:1000]
np.random.shuffle(combined_train)
X5 = X_train[:1000]
y_train5 = y_train[:1000]
matdic = {'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5}
savemat("input/subsets.mat", matdic)

print(" *** SVM *** ") 

svm_clf = SVC(kernel='rbf', decision_function_shape='ovr')
svm_clf.fit(X1, y_train1.ravel())

svm_pred = svm_clf.predict(X1)
print(f"SVM Error on X1: {1 - accuracy_score(y_train1, svm_pred)}")
svm_pred = svm_clf.predict(X2)
print(f"SVM Error on X2: {1 - accuracy_score(y_train2, svm_pred)}")
svm_pred = svm_clf.predict(X3)
print(f"SVM Error on X3: {1 - accuracy_score(y_train3, svm_pred)}")
svm_pred = svm_clf.predict(X4)
print(f"SVM Error on X4: {1 - accuracy_score(y_train4, svm_pred)}")
svm_pred = svm_clf.predict(X5)
print(f"SVM Error on X5: {1 - accuracy_score(y_train5, svm_pred)}")
svm_pred = svm_clf.predict(X_test)
print(f"SVM Error on X_test: {1 - accuracy_score(y_test, svm_pred)}")

print(" *** KNN *** ") 

knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X2, y_train2.ravel())

knn_pred = knn_clf.predict(X1)
print(f"KNN Error on X1: {1 - accuracy_score(y_train1, knn_pred)}")
knn_pred = knn_clf.predict(X2)
print(f"KNN Error on X2: {1 - accuracy_score(y_train2, knn_pred)}")
knn_pred = knn_clf.predict(X3)
print(f"KNN Error on X3: {1 - accuracy_score(y_train3, knn_pred)}")
knn_pred = knn_clf.predict(X4)
print(f"KNN Error on X4: {1 - accuracy_score(y_train4, knn_pred)}")
knn_pred = knn_clf.predict(X5)
print(f"KNN Error on X5: {1 - accuracy_score(y_train5, knn_pred)}")
knn_pred = knn_clf.predict(X_test)
print(f"KNN Error on X_test: {1 - accuracy_score(y_test, knn_pred)}")

print(" *** Logistic Regression *** ") 

lr_clf = LogisticRegression()
lr_clf.fit(X3, y_train3.ravel())

lr_pred = lr_clf.predict(X1)
print(f"LR Error on X1: {1 - accuracy_score(y_train1, lr_pred)}")
lr_pred = lr_clf.predict(X2)
print(f"LR Error on X2: {1 - accuracy_score(y_train2, lr_pred)}")
lr_pred = lr_clf.predict(X3)
print(f"LR Error on X3: {1 - accuracy_score(y_train3, lr_pred)}")
lr_pred = lr_clf.predict(X4)
print(f"LR Error on X4: {1 - accuracy_score(y_train4, lr_pred)}")
lr_pred = lr_clf.predict(X5)
print(f"LR Error on X5: {1 - accuracy_score(y_train5, lr_pred)}")
lr_pred = lr_clf.predict(X_test)
print(f"LR Error on X_test: {1 - accuracy_score(y_test, lr_pred)}")

print(" *** Decision Tree *** ") 

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X4, y_train4.ravel())

dt_pred = dt_clf.predict(X1)
print(f"DT Error on X1: {1 - accuracy_score(y_train1, dt_pred)}")
dt_pred = dt_clf.predict(X2)
print(f"DT Error on X2: {1 - accuracy_score(y_train2, dt_pred)}")
dt_pred = dt_clf.predict(X3)
print(f"DT Error on X3: {1 - accuracy_score(y_train3, dt_pred)}")
dt_pred = dt_clf.predict(X4)
print(f"DT Error on X4: {1 - accuracy_score(y_train4, dt_pred)}")
dt_pred = dt_clf.predict(X5)
print(f"DT Error on X5: {1 - accuracy_score(y_train5, dt_pred)}")
dt_pred = dt_clf.predict(X_test)
print(f"DT Error on X_test: {1 - accuracy_score(y_test, dt_pred)}")

print(" *** Random Forest *** ")

rf_clf = RandomForestClassifier(n_estimators=85)
rf_clf.fit(X5, y_train5.ravel())

rf_pred = rf_clf.predict(X1)
print(f"RF Error on X1: {1 - accuracy_score(y_train1, rf_pred)}")
rf_pred = rf_clf.predict(X2)
print(f"RF Error on X2: {1 - accuracy_score(y_train2, rf_pred)}")
rf_pred = rf_clf.predict(X3)
print(f"RF Error on X3: {1 - accuracy_score(y_train3, rf_pred)}")
rf_pred = rf_clf.predict(X4)
print(f"RF Error on X4: {1 - accuracy_score(y_train4, rf_pred)}")
rf_pred = rf_clf.predict(X5)
print(f"RF Error on X5: {1 - accuracy_score(y_train5, rf_pred)}")
rf_pred = rf_clf.predict(X_test)
print(f"RF Error on X_test: {1 - accuracy_score(y_test, rf_pred)}")

print(" *** Majority Voting *** ")
voting_clf = VotingClassifier(estimators=[
    ('svm', svm_clf),
    ('knn', knn_clf),
    ('lr', lr_clf),
    ('dt', dt_clf),
    ('rf', rf_clf)])
voting_clf.fit(X_train, y_train.ravel())
v_pred = voting_clf.predict(X_test)
print(f"MV Error on X_test: {1 - accuracy_score(y_test, v_pred)}")