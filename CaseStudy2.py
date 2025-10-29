import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype('float32')
y = mnist.target.astype('int')

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

pca = PCA(n_components=60)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

start = time.time()
svm_model = SVC(kernel='rbf', C=5, gamma=0.01)
svm_model.fit(X_train_pca, y_train)
svm_pred = svm_model.predict(X_test_pca)
svm_time = time.time() - start

svm_acc = accuracy_score(y_test, svm_pred)
print("SVM Accuracy:", svm_acc)
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
print("SVM Classification Report:\n", classification_report(y_test, svm_pred))
print("SVM Training Time:", svm_time, "seconds")

start = time.time()
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_pca, y_train)
knn_pred = knn_model.predict(X_test_pca)
knn_time = time.time() - start

knn_acc = accuracy_score(y_test, knn_pred)
print("k-NN Accuracy:", knn_acc)
print("k-NN Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
print("k-NN Classification Report:\n", classification_report(y_test, knn_pred))
print("k-NN Training Time:", knn_time, "seconds")