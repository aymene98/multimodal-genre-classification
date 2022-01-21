from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
from joblib import dump


# load data
def load_audio_data():
    filehandler = open("dataset.txt","rb")
    dataset = pickle.load(filehandler)
    train_audios, val_audios, test_audios = [],[],[]
    train_labels, val_labels, test_labels = [],[],[]
    
    for instance in dataset:
        if instance['instance_set'] == 'train':
            train_audios.append(instance['audio'])
            train_labels.append(instance['label'])
        elif instance['instance_set'] == 'val':
            val_audios.append(instance['audio'])
            val_labels.append(instance['label'])
        else:
            test_audios.append(instance['audio'])
            test_labels.append(instance['label'])
        
    return np.asarray(train_audios), np.asarray(val_audios), np.asarray(test_audios), np.asarray(train_labels), np.asarray(val_labels), np.asarray(test_labels)


def main():
    print('Loading data ...')
    train_audios, val_audios, test_audios, train_labels, val_labels, test_labels = load_audio_data()

    print(train_audios.shape)
    print(val_audios.shape)
    print(test_audios.shape)
    print(train_labels.shape)
    print(val_labels.shape)

    X = np.concatenate((train_audios, val_audios), axis=0)
    y = np.concatenate((train_labels, val_labels), axis=0)

    print('Training SVM ...')
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', max_iter=1000))
    clf.fit(X, y)
    print("SVM test score : ",clf.score(test_audios, test_labels))
    #dump(clf, './models/SVM_audio_proba.joblib') 

    print('Training MLP ...')
    clf = make_pipeline(StandardScaler(), MLPClassifier(max_iter=10))
    clf.fit(X, y)
    print("MLP test score : ", clf.score(test_audios, test_labels))
    #dump(clf, './models/MLP_audio.joblib') 

#main()

# audio MFCC:
# 40.98% pour 1000 itérations SVM
# 46.24% pour 10 itérations MLP