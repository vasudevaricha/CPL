# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [code] Cell 1
# !pip install numba  # (magic command commented out)

# %% [code] Cell 2
# !pip install pandas  # (magic command commented out)

# %% [code] Cell 3
# !pip install scikit-learn  # (magic command commented out)

# %% [code] Cell 4
# !pip install matplotlib  # (magic command commented out)

# %% [code] Cell 5
# !nvidia-smi  # (magic command commented out)

# %% [code] Cell 6
import pandas as pd
import numpy as np
from numba import cuda
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# %% [code] Cell 7
from google.colab import files
uploaded = files.upload()

# %% [code] Cell 8
import pandas as pd

df = pd.read_csv("Titanic-Dataset.csv")

# %% [code] Cell 9
#eda
print(df.head())
print(df.info())
print(df.describe())

# %% [code] Cell 10
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

le_sex = LabelEncoder()
le_embarked = LabelEncoder()

df['Sex'] = le_sex.fit_transform(df['Sex'])
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

# %% [code] Cell 11
X = df.drop('Survived', axis=1).values.astype(np.float32)
y = df['Survived'].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %% [code] Cell 12
@cuda.jit(device=True)
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# %% [code] Cell 13
@cuda.jit
def predict_kernel(X, weights, bias, preds):
    i = cuda.grid(1)

    if i < X.shape[0]:
        z = 0.0

        for j in range(X.shape[1]):
            z += X[i, j] * weights[j]

        z += bias
        preds[i] = sigmoid(z)

# %% [code] Cell 14
@cuda.jit
def gradient_kernel(X, y, weights, bias, grad_w, grad_b):
    i = cuda.grid(1)

    if i < X.shape[0]:
        z = 0.0

        for j in range(X.shape[1]):
            z += X[i, j] * weights[j]

        z += bias

        pred = sigmoid(z)
        error = pred - y[i]

        for j in range(X.shape[1]):
            cuda.atomic.add(grad_w, j, error * X[i, j])

        cuda.atomic.add(grad_b, 0, error)

# %% [code] Cell 15
n_features = X_train.shape[1]

weights = np.zeros(n_features, dtype=np.float32)
bias = np.float32(0.0)

d_X = cuda.to_device(X_train)
d_y = cuda.to_device(y_train)
d_weights = cuda.to_device(weights)

threads_per_block = 256
blocks = (X_train.shape[0] + threads_per_block - 1) // threads_per_block

lr = 0.01
epochs = 100



# %% [code] Cell 16
for epoch in range(epochs):

    grad_w = np.zeros(n_features, dtype=np.float32)
    grad_b = np.zeros(1, dtype=np.float32)

    d_grad_w = cuda.to_device(grad_w)
    d_grad_b = cuda.to_device(grad_b)

    gradient_kernel[blocks, threads_per_block](
        d_X, d_y, d_weights, bias, d_grad_w, d_grad_b
    )

    grad_w = d_grad_w.copy_to_host() / X_train.shape[0]
    grad_b = d_grad_b.copy_to_host()[0] / X_train.shape[0]

    weights -= lr * grad_w
    bias -= lr * grad_b

    d_weights = cuda.to_device(weights)

# %% [code] Cell 17
d_Xtest = cuda.to_device(X_test)

preds = np.zeros(X_test.shape[0], dtype=np.float32)
d_preds = cuda.to_device(preds)

blocks_test = (X_test.shape[0] + threads_per_block - 1) // threads_per_block

predict_kernel[blocks_test, threads_per_block](d_Xtest, d_weights, bias, d_preds)

preds = d_preds.copy_to_host()

# %% [code] Cell 18
pred_labels = (preds >= 0.5).astype(np.int32)

accuracy = np.mean(pred_labels == y_test)

print("Accuracy:", accuracy)

# %% [code] Cell 19
tp = np.sum((pred_labels == 1) & (y_test == 1))
tn = np.sum((pred_labels == 0) & (y_test == 0))
fp = np.sum((pred_labels == 1) & (y_test == 0))
fn = np.sum((pred_labels == 0) & (y_test == 1))

print("Confusion Matrix:")
print(f"TP: {tp}, FP: {fp}")
print(f"FN: {fn}, TN: {tn}")

# %% [code] Cell 20
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

plt.show()

# %% [code] Cell 21
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# %% [code] Cell 22
scaler.fit(X_train)

# %% [code] Cell 23
samples_scaled = scaler.transform(samples).astype(np.float32)

# %% [code] Cell 24
samples = samples[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
samples_scaled = scaler.transform(samples).astype(np.float32)

# %% [code] Cell 25
# ===== SAMPLE DATA =====
samples = pd.DataFrame([
    [3, 'male', 22, 1, 0, 7.25, 'S'],
    [1, 'female', 38, 1, 0, 71.28, 'C'],
    [2, 'female', 26, 0, 0, 13.00, 'S']
], columns=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])

samples['Sex'] = le_sex.transform(samples['Sex'])
samples['Embarked'] = le_embarked.transform(samples['Embarked'])


# ===== SCALING =====
samples_scaled = scaler.transform(samples).astype(np.float32)


# ===== MOVE TO GPU =====
d_samples = cuda.to_device(samples_scaled)

preds = np.zeros(samples.shape[0], dtype=np.float32)
d_preds = cuda.to_device(preds)


# ===== CUDA PREDICTION =====
threads_per_block = 256
blocks = (samples.shape[0] + threads_per_block - 1) // threads_per_block

predict_kernel[blocks, threads_per_block](
    d_samples, d_weights, bias, d_preds
)


# ===== RESULTS =====
preds = d_preds.copy_to_host()

samples['Survival_Prob'] = preds
samples['Prediction'] = [
    'Survived' if p > 0.5 else 'Not Survived' for p in preds
]


print("===== PREDICTION RESULTS =====")
samples
