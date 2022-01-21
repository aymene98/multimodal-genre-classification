from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
from joblib import dump

# load data
def load_image_data():
    filehandler = open("dataset.txt","rb")
    dataset = pickle.load(filehandler)
    train_img, val_img, test_img = [],[],[]
    train_labels, val_labels, test_labels = [],[],[]
    
    for instance in dataset:
        if instance['instance_set'] == 'train':
            train_img.append(instance['image'])
            train_labels.append(instance['label'])
        elif instance['instance_set'] == 'val':
            val_img.append(instance['image'])
            val_labels.append(instance['label'])
        else:
            test_img.append(instance['image'])
            test_labels.append(instance['label'])
        
    return np.asarray(train_img), np.asarray(val_img), np.asarray(test_img), np.asarray(train_labels), np.asarray(val_labels), np.asarray(test_labels)

def load_hog_data():
    filehandler = open("dataset.txt","rb")
    dataset = pickle.load(filehandler)
    train_img, val_img, test_img = [],[],[]
    train_labels, val_labels, test_labels = [],[],[]
    
    for instance in dataset:
        if instance['instance_set'] == 'train':
            train_img.append(instance['hog'])
            train_labels.append(instance['label'])
        elif instance['instance_set'] == 'val':
            val_img.append(instance['hog'])
            val_labels.append(instance['label'])
        else:
            test_img.append(instance['hog'])
            test_labels.append(instance['label'])
        
    return np.asarray(train_img), np.asarray(val_img), np.asarray(test_img), np.asarray(train_labels), np.asarray(val_labels), np.asarray(test_labels)

def main():
    print('Loading data ...')

    #train_img, val_img, test_img, train_labels, val_labels, test_labels = load_image_data()
    train_img, val_img, test_img, train_labels, val_labels, test_labels = load_hog_data()
    print(train_img.shape)
    print(val_img.shape)
    print(test_img.shape)
    print(train_labels.shape)
    print(val_labels.shape)
    print(test_labels.shape)

    X = np.concatenate((train_img, val_img), axis=0)
    shape = X.shape[0]
    X = np.reshape(X, (shape, -1))
    y = np.concatenate((train_labels, val_labels), axis=0)


    print('Training SVM ...')
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', max_iter=10))
    clf.fit(X, y)
    print("SVM test score : ",clf.score(test_img, test_labels))
    #dump(clf, './models/SVM_img_proba.joblib')

    
    print('Training MLP ...')
    clf = make_pipeline(StandardScaler(), MLPClassifier(max_iter=10))
    clf.fit(X, y)
    print("MLP test score : ", clf.score(test_img, test_labels))
    #dump(clf, './models/MLP_img.joblib') 

#main()

# 15 classes
# images HOG:
# 18.26% pour 1000 itérations SVM
# 17.60% pour 10 itérations MLP