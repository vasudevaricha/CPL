# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [code] Cell 1
# !nvidia-smi  # (magic command commented out)


# %% [code] Cell 2
# !pip install -q kaggle numba  # (magic command commented out)


# %% [code] Cell 3
from google.colab import files
files.upload()   # Upload kaggle.json


# %% [code] Cell 4
import os

os.makedirs('/root/.kaggle', exist_ok=True)
# !cp kaggle.json /root/.kaggle/  # (magic command commented out)
# !chmod 600 /root/.kaggle/kaggle.json  # (magic command commented out)


# %% [code] Cell 5
# !kaggle datasets download -d omkargurav/face-mask-dataset  # (magic command commented out)


# %% [code] Cell 6
# !ls  # (magic command commented out)


# %% [code] Cell 7
# !unzip -q face-mask-dataset.zip  # (magic command commented out)


# %% [code] Cell 8
import numpy as np
from PIL import Image
import os

def load_dataset():
    images = []
    labels = []

    base_path = "data"   # ← IMPORTANT

    for label, folder in enumerate(['without_mask', 'with_mask']):
        folder_path = os.path.join(base_path, folder)

        print("Loading from:", folder_path)

        if not os.path.exists(folder_path):
            print("Folder not found:", folder_path)
            continue

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)

            try:
                img = Image.open(img_path).convert("L")
                img = img.resize((32,32))
                img = np.array(img).astype(np.float32) / 255.0

                images.append(img)
                labels.append(label)
            except:
                continue

    return np.array(images), np.array(labels)


X, y = load_dataset()

print("Dataset Loaded ✅")
print("Images shape:", X.shape)
print("Labels shape:", y.shape)


# %% [code] Cell 9
from numba import cuda
import math

# ----------------------------
# CUDA Convolution Kernel
# ----------------------------
@cuda.jit
def conv2d_kernel(input_img, filters, output):
    x, y, f = cuda.grid(3)

    if (x < output.shape[0] and
        y < output.shape[1] and
        f < output.shape[2]):

        sum_val = 0.0

        for i in range(3):
            for j in range(3):
                sum_val += input_img[x+i, y+j] * filters[f, i, j]

        output[x, y, f] = sum_val


# ----------------------------
# CUDA Dense Kernel
# ----------------------------
@cuda.jit
def dense_kernel(input_vec, weights, bias, output):
    idx = cuda.grid(1)

    if idx < output.shape[0]:
        tmp = 0.0
        for i in range(input_vec.shape[0]):
            tmp += input_vec[i] * weights[i, idx]
        output[idx] = tmp + bias[idx]


def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -(y_true * np.log(y_pred) +
             (1 - y_true) * np.log(1 - y_pred))


def predict_label(prob):
    return "Mask" if prob > 0.5 else "No Mask"



# %% [code] Cell 10
def init_model(flatten_size):
    params = {}

    params['conv'] = np.random.randn(4,3,3).astype(np.float32) * 0.01
    params['dense_w'] = np.random.randn(flatten_size,1).astype(np.float32) * 0.01
    params['dense_b'] = np.zeros(1).astype(np.float32)

    return params

flatten_size = (32-2)*(32-2)*4
params = init_model(flatten_size)


# %% [code] Cell 11
def forward_pass(image, params):

    d_image = cuda.to_device(image)
    d_filters = cuda.to_device(params['conv'])

    out_h = 30
    out_w = 30
    out_c = 4

    output = np.zeros((out_h, out_w, out_c), dtype=np.float32)
    d_output = cuda.to_device(output)

    threads = (8,8,4)
    blocks = (math.ceil(out_h/8), math.ceil(out_w/8), 1)

    conv2d_kernel[blocks, threads](d_image, d_filters, d_output)

    conv_out = d_output.copy_to_host()
    conv_out = relu(conv_out)

    flat = conv_out.flatten().astype(np.float32)

    d_flat = cuda.to_device(flat)
    d_weights = cuda.to_device(params['dense_w'])
    d_bias = cuda.to_device(params['dense_b'])

    dense_out = np.zeros(1, dtype=np.float32)
    d_dense_out = cuda.to_device(dense_out)

    dense_kernel[1,32](d_flat, d_weights, d_bias, d_dense_out)

    final = sigmoid(d_dense_out.copy_to_host())

    return final


# %% [code] Cell 12
prediction = forward_pass(X[0], params)

prob = prediction[0]
predicted_class = 1 if prob > 0.5 else 0

print("Prediction Probability:", prob)
print("Predicted Class:", predict_label(prob))
print("Actual Class:", "Mask" if y[0] == 1 else "No Mask")


# %% [code] Cell 13
total_loss = 0
correct = 0
num_samples = 100   # evaluate first 100 images (safe for Colab GPU)

for i in range(num_samples):

    pred = forward_pass(X[i], params)
    prob = pred[0]

    # Compute loss
    loss = binary_cross_entropy(y[i], prob)
    total_loss += loss

    # Compute accuracy
    predicted_class = 1 if prob > 0.5 else 0
    if predicted_class == y[i]:
        correct += 1

avg_loss = total_loss / num_samples
accuracy = correct / num_samples

print("\n===== Model Evaluation =====")
print("Average Loss:", avg_loss)
print("Accuracy:", accuracy)

