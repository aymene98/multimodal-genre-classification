from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import tensorflow as tf

from tensorflow.keras import datasets, layers, models


# loading data

X_train = np.load('./data/img_train_data.npy')
X_val = np.load('./data/img_val_data.npy')
X_test = np.load('./data/img_test_data.npy')

y_train = np.load('./data/y_tain_data.npy')
y_val = np.load('./data/y_val_data.npy')
y_test = np.load('./data/y_test_data.npy')

"""X = np.concatenate((X_train, X_val), axis=0)
y = np.concatenate((y_tain, y_val), axis=0)
shape = X.shape
X = X.reshape((shape[0], -1))"""

"""shape = X_test.shape
X_test = X_test.reshape((shape[0], -1))"""

#clf = make_pipeline(StandardScaler(), SVC(gamma='auto', max_iter=1000))
#clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
#clf = make_pipeline(StandardScaler(), DecisionTreeClassifier())
clf = make_pipeline(StandardScaler(), MLPClassifier(max_iter=1))

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(15))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train[:1000], y_train[:1000], epochs=10, validation_data=(X_val, y_val), batch_size=100)

test_loss, test_acc = model.evaluate(X_test,  y_test)

print(test_loss, test_acc)

"""
print("Fitting classifier ...")
clf.fit(X, y)

print("Scoring classifier ...")
score = clf.score(X_test, y_test)
print("Score : ",score)

# 15 classes
# images HOG:
# 18% pour 1000 itérations SVM
# 18% pour 10 itérations MLP

# 15 classes
# images :
# 17.78% pour 10 itérations CNN (colab)

# audio MFCC:
# 41% pour 1000 itérations SVM
# 47% pour 10 itérations MLP
# 
# """