import numpy as np
from sklearn.datasets import load_digits

dataset = load_digits()

# dataset information
print(dataset.data)
print(dataset.target)

print(dataset.data.shape)
print(dataset.images.shape)

dataimagelength = len(dataset.images)
print(dataimagelength)

# visualize the dataset

import matplotlib.pyplot as plt

n = 9
plt.figure(figsize=(4,4))
plt.imshow(dataset.images[n], cmap='gray')
plt.title(f"Digit: {dataset.target[n]}")
plt.axis('off')
plt.show()

x = dataset.images.reshape((dataimagelength, -1))
y = dataset.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0
)

print(x_train.shape)
print(x_test.shape)

# training
from sklearn import svm

model = svm.SVC()
model.fit(x_train, y_train)

# predicting what the digit is from test data
n = 13

result = model.predict(dataset.images[n].reshape((1, -1)))

plt.imshow(
    dataset.images[n],
    cmap=plt.cm.gray_r,
    interpolation="nearest"
)

print("result", result)
print('\n')

plt.axis('off')
plt.title("%i" % result[0])
plt.show()

# prediction of the model
y_pred = model.predict(x_test)

print(
    np.concatenate(
        (
            y_pred.reshape(len(y_pred), 1),
            y_test.reshape(len(y_test), 1)
        ),
        axis=1
    )
)

# evaluate model - accuracy score
from sklearn.metrics import accuracy_score

print(
    "accuracy of the model : {0}%".format(
        accuracy_score(y_test, y_pred) * 100
    )
)