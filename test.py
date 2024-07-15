# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 01:05:08 2024

@author: nguye
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Đọc dữ liệu từ tệp Iris.csv
file_path = 'C:/Users/nguye/Downloads/Iris.csv'
iris_data = pd.read_csv(file_path)

# Hiển thị dữ liệu
print(iris_data.head())

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X = iris_data.drop(columns=['Id', 'Species'])
y = iris_data['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Xây dựng mô hình KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Dự đoán trên tập huấn luyện và tập kiểm tra
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Đánh giá mô hình
train_classification_report = classification_report(y_train, y_train_pred)
test_classification_report = classification_report(y_test, y_test_pred)
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)

# Hiển thị confusion matrix bằng heatmap
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
sns.heatmap(train_conf_matrix, annot=True, fmt="d", cmap="Oranges", xticklabels=iris_data['Species'].unique(), yticklabels=iris_data['Species'].unique())
plt.xlabel('Predicted')
plt.ylabel('True Label')
plt.title('Train Confusion Matrix')

plt.subplot(1, 2, 2)
sns.heatmap(test_conf_matrix, annot=True, fmt="d", cmap="Oranges", xticklabels=iris_data['Species'].unique(), yticklabels=iris_data['Species'].unique())
plt.xlabel('Predicted')
plt.ylabel('True Label')
plt.title('Test Confusion Matrix')

plt.show()

print("Train Classification Report:")
print(train_classification_report)
print("Test Classification Report:")
print(test_classification_report)


