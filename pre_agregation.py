from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
from text_classifier import load_text_data
from image_classifier import load_hog_data
from audio_classifier import load_audio_data

def filter_data(train_x, val_x, test_x):
    train_indeces, val_indeces, test_indeces = [],[],[]
    train_text, val_text, test_text = [],[],[]
    for i, instance in enumerate(train_x):
        if np.any(instance != None):
            train_indeces.append(i)
            train_text.append(instance)
            
    for i, instance in enumerate(val_x):
        if np.any(instance != None):
            val_indeces.append(i)
            val_text.append(instance)
            
    for i, instance in enumerate(test_x):
        if np.any(instance != None):
            test_indeces.append(i)
            test_text.append(instance)
            
    return np.asarray(train_indeces), np.asarray(val_indeces), np.asarray(test_indeces), np.asarray(train_text), np.asarray(val_text), np.asarray(test_text)

def main(svm = True):
    print('loading data ...')
    train_text, val_text, test_text, _, _, _ = load_text_data()
    train_img, val_img, test_img, _, _, _ = load_hog_data()
    train_audios, val_audios, test_audios, train_labels, val_labels, test_labels = load_audio_data()
    
    print("preprocessing ...")
    train_indeces, val_indeces, test_indeces, train_text, val_text, test_text = filter_data(train_text, val_text, test_text)
    
    train_X = np.concatenate((train_text, train_img[train_indeces,:], train_audios[train_indeces,:]), axis=1)
    val_X = np.concatenate((val_text, val_img[val_indeces,:], val_audios[val_indeces,:]), axis=1)
    test_X = np.concatenate((test_text, test_img[test_indeces,:], test_audios[test_indeces,:]), axis=1)
    
    train_y = train_labels[train_indeces]
    val_y = val_labels[val_indeces]
    test_y = test_labels[test_indeces]
    
    X = np.concatenate((train_X, val_X), axis=0)
    y = np.concatenate((train_y, val_y), axis=0)
    
    if svm:
        print('Training SVM ...')
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto', max_iter=100))
        clf.fit(X, y)
        print("SVM test score : ",clf.score(test_X, test_y))
    
    else:
        print('Training MLP ...')
        clf = make_pipeline(StandardScaler(), MLPClassifier(max_iter=50))
        clf.fit(X, y)
        print("MLP test score : ", clf.score(test_X, test_y))
    
main(svm=False)
# 22.91% pour 100 itérations SVM
# 42% pour 10 itérations MLP
