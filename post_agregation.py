from text_classifier import load_text_data
from image_classifier import load_hog_data
from audio_classifier import load_audio_data
from joblib import load
import numpy as np

def predict_text(clf, data, proba=False):
    labels = []
    for instance in data:
        if np.any(instance == None):
            if not proba: 
                labels.append(-1)
            else:
                labels.append(np.asarray(None))
        else:
            if not proba: 
                labels.append(clf.predict(instance.reshape(1,-1))[0])        
            else:
                labels.append(clf.predict_proba(instance.reshape(1,-1))[0])        
    return labels

def agregate_predictions(audio, hog, text):
    labels = []
    for i,v in enumerate(audio):
        temp = [0]*15
        temp[v]+=1
        temp[hog[i]]+=1
        if text[i]!=-1 and text[i] is not None: 
            temp[text[i]]+=1
        labels.append(temp.index(max(temp)))
    return labels

def weighted_predictions(audio, hog, text, weights):
    labels = []
    for i,v in enumerate(audio):
        temp = np.asarray(v) * weights[0]
        temp+= np.asarray(hog[i]) * weights[1]
        if np.any(text[i] != None): 
            temp += np.asarray(text[i]) * weights[2]
        labels.append(np.argmax(temp))
    return labels

def accuracy(predictions, y):
    trues = 0
    for i,prediction in enumerate(predictions):
        if prediction==y[i]:
            trues +=1
    return trues/len(y)

def evaluate_majorite(svm=True):
    print("loading data ...")
    _, _, test_hog, _, _, test_labels = load_hog_data()
    _, _, test_text, _, _, _ = load_text_data()
    _, _, test_audios, _, _, _ = load_audio_data()
    
    print("loading models ...")
    if svm:
        clf_audio = load('./models/SVM_audio.joblib')
        clf_hog = load('./models/SVM_hog.joblib')
        clf_text = load('./models/SVM_text.joblib')
    else:
        clf_audio = load('./models/MLP_audio.joblib')
        clf_hog = load('./models/MLP_hog.joblib')
        clf_text = load('./models/MLP_text.joblib')
    
    print('predicting audio')
    predictions_audio = clf_audio.predict(test_audios).tolist()
    print('predicting hog')
    predictions_hog = clf_hog.predict(test_hog).tolist()
    print('predicting text')
    predictions_text = predict_text(clf_text,test_text)
    
    labels = agregate_predictions(predictions_audio, predictions_hog, predictions_text)
    
    acc = accuracy(labels, test_labels)
    print("Accracy : %.4f"%acc)
    
def evaluate_weighted(svm=True):
    print("loading data ...")
    _, _, test_hog, _, _, test_labels = load_hog_data()
    _, _, test_text, _, _, _ = load_text_data()
    _, _, test_audios, _, _, _ = load_audio_data()
    
    print("loading models ...")
    if svm:
        clf_audio = load('./models/SVM_audio.joblib')
        clf_hog = load('./models/SVM_hog.joblib')
        clf_text = load('./models/SVM_text.joblib')
        weights = (0.51, 0.225, 0.268)
    else:
        clf_audio = load('./models/MLP_audio.joblib')
        clf_hog = load('./models/MLP_hog.joblib')
        clf_text = load('./models/MLP_text.joblib')
        weights = (0.46, 0.175, 0.364)
    
    print('predicting audio')
    predictions_audio = clf_audio.predict_proba(test_audios).tolist()
    print('predicting hog')
    predictions_hog = clf_hog.predict_proba(test_hog).tolist()
    print('predicting text')
    predictions_text = predict_text(clf_text,test_text, proba=True)
    
    labels = weighted_predictions(predictions_audio, predictions_hog, predictions_text, weights)
    
    acc = accuracy(labels, test_labels)
    print("Accracy : %.4f"%acc)
    
evaluate_weighted(svm=False)
#evaluate_majorite()
# classe majoritaire :
# 32.33% SVM
# 37.19% MLP

# classe weighted :
# ...% SVM 
# 44.76% MLP