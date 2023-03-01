import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import pickle


def get_features():
    train_data = np.load('train_data.npy', allow_pickle=True)
    test_data = np.load('test_data.npy', allow_pickle=True)

    train_pairs = []
    test_pairs = []

    for ind in range(len(train_data)):
        img = train_data[ind][0]
        label = train_data[ind][1]
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                            visualize=True, multichannel=True)
        train_pairs.append([fd, label])

    for ind in range(len(test_data)):
        img = test_data[ind][0]
        label = test_data[ind][1]
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                            visualize=True, multichannel=True)
        test_pairs.append([fd, label])

    np.save('train_features.npy', train_pairs)
    np.save('test_features.npy', test_pairs)


# get_features()

train_features = np.load('train_features.npy', allow_pickle=True)
test_features = np.load('test_features.npy', allow_pickle=True)

X_train, y_train, X_test, y_test = [], [], [], []
for i in range(len(train_features)):
    X_train.append(train_features[i][0])
    y_train.append(train_features[i][1])

for i in range(len(test_features)):
    X_test.append(test_features[i][0])
    y_test.append(test_features[i][1])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Models
# --------->
# without_kernal = SVC().fit(X_train, y_train)
# pickle.dump(without_kernal, open('linear', 'wb'))
without_kernal = pickle.load(open('linear', 'rb'))
prediction = without_kernal.predict(X_test)
print("Accuracy of without_kernal: "+str(float(accuracy_score(y_test, prediction)) * 100))

# svm_model_linear_ovo = SVC(kernel='poly', degree=4, C=1000000).fit(X_train, y_train)
# pickle.dump(svm_model_linear_ovo, open('ovo', 'wb'))
svm_model_linear_ovo = pickle.load(open('ovo', 'rb'))
prediction = svm_model_linear_ovo.predict(X_test)
print("Accuracy of ovo: "+str(float(accuracy_score(y_test, prediction)) * 100))

# svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='poly', degree=4, C=1000000)).fit(X_train, y_train)
# pickle.dump(svm_model_linear_ovr, open('ovr', 'wb'))
svm_model_linear_ovr = pickle.load(open('ovr', 'rb'))
prediction = svm_model_linear_ovr.predict(X_test)
print("Accuracy of ovr: "+str(float(accuracy_score(y_test, prediction)) * 100))
