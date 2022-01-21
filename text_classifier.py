from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump
from sqlalchemy import true



# load data
def load_text_data():
    filehandler = open("dataset.txt","rb")
    dataset = pickle.load(filehandler)
    train_text, val_text, test_text = [],[],[]
    train_labels, val_labels, test_labels = [],[],[]
    
    for instance in dataset:
        if instance['instance_set'] == 'train':
            train_text.append(instance['bag_of_words'])
            train_labels.append(instance['label'])
        elif instance['instance_set'] == 'val':
            val_text.append(instance['bag_of_words'])
            val_labels.append(instance['label'])
        else:
            test_text.append(instance['bag_of_words'])
            test_labels.append(instance['label'])
        
    return np.asarray(train_text), np.asarray(val_text), np.asarray(test_text), np.asarray(train_labels), np.asarray(val_labels), np.asarray(test_labels)

def filter_text_data(train_x, val_x, test_x, train_y, val_y, test_y):
    # filtering None values ... 
    train_text, val_text, test_text = [],[],[]
    train_labels, val_labels, test_labels = [],[],[]
    for i, instance in enumerate(train_x):
        #print(instance)
        if np.any(instance != None):
            train_text.append(instance)
            train_labels.append(train_y[i])
            
    for i, instance in enumerate(val_x):
        if np.any(instance != None):
            val_text.append(instance)
            val_labels.append(val_y[i])
            
    for i, instance in enumerate(test_x):
        if np.any(instance != None):
            test_text.append(instance)
            test_labels.append(test_y[i])
            
    return np.asarray(train_text), np.asarray(val_text), np.asarray(test_text), np.asarray(train_labels), np.asarray(val_labels), np.asarray(test_labels)

def main():
    print('Loading data ...')

    train_text, val_text, test_text, train_labels, val_labels, test_labels = load_text_data()
    train_text, val_text, test_text, train_labels, val_labels, test_labels = filter_text_data(train_text, val_text, test_text, train_labels, val_labels, test_labels)

    print(train_text.shape)
    print(val_text.shape)
    print(test_text.shape)
    print(train_labels.shape)
    print(val_labels.shape)
    print(test_labels.shape)

    X = np.concatenate((train_text, val_text), axis=0)
    y = np.concatenate((train_labels, val_labels), axis=0)
    print(X.shape)

    print('Training SVM ...')
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', max_iter=100))
    clf.fit(X, y)
    print("SVM test score : ",clf.score(test_text, test_labels))
    #dump(clf, './models/SVM_text_proba.joblib') 

    print('Training MLP ...')
    clf = make_pipeline(StandardScaler(), MLPClassifier(max_iter=10))
    clf.fit(X, y)
    print("MLP test score : ", clf.score(test_text, test_labels))
    #dump(clf, './models/MLP_text.joblib')

#main()
# 21.68% pour 100 itérations SVM
# 36.55% pour 10 itérations MLP