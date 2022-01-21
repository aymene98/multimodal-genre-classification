from msdi_io import *
import numpy as np
from skimage.feature import hog
from librosa.feature import mfcc
from os import listdir
import json, pickle

_msdi_path = 'msdi'
orientation_hog_img = 8
P=Q=3

def contructing_img_feature(img):
    hog_vector, _ = hog(img, orientations=orientation_hog_img, pixels_per_cell=(img.shape[0]//P, img.shape[1]//Q),
        cells_per_block=(1, 1), visualize=True, multichannel=True)
    return hog_vector
    
def contructing_tram_feature(tram):
    return np.sum(tram, axis=0)

        

def building_data():
    labels = get_label_list()
    msdi = get_msdi_dataframe(_msdi_path)
    with open('bag_of_words.json', 'r') as fp:
        bag_of_words = json.load(fp)
    dataset = []
    for index in range(len(msdi)):
        entry = msdi.loc[index] # index in the data file
        set = get_set(entry) # train/val/test
        track_id = get_track_id(entry) # track ID
        
        tram = load_mfcc(entry, _msdi_path) # load mfcc
        tram_feature = contructing_tram_feature(tram) # audio features
        
        img = load_img(entry, _msdi_path) # get image
        img_feature = contructing_img_feature(img) # img features
        
        entry_label = get_label(entry) # instance label
        
        # bag_of_words in the dict. I built it before 
        
        info = {
            'id' : track_id,
            'instance_set' : set,
            'label' : labels.index(entry_label),
            'audio' : tram_feature,
            'image' : img,
            'hog' : img_feature,
            'bag_of_words': np.asarray(bag_of_words[track_id]) if track_id in bag_of_words.keys() else None
        }
        
        dataset.append(info)
    return dataset
        
#print(load_mfcc_dicts()["mfcc_A.npz"].keys())

dataset = building_data()
#filehandler = open("dataset.txt","wb")
#pickle.dump(dataset,filehandler)