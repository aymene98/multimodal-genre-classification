"""import json
import numpy as np

f = open('./msdi/msx_lyrics_genre.txt')

comment = {}
words_in_vocab = 0
for line in f.readlines():
    id = line.split()[0]
    words_freq = [(int(x.split(":")[0]),int(x.split(":")[1])) for x in line.split()[2:]]
    feature = [0]*5000
    for t in words_freq:
        index, value = t[0], t[1]
        feature[index-1] = value
    comment[id] = feature
print(type(np.asarray(comment["TRAACER128F4290F96"][:20])))
# 5000 mots dans le vocabulaire

with open('bag_of_words.json', 'w') as fp:
    json.dump(comment, fp)
"""
import json
with open('bag_of_words.json', 'r') as fp:
    temp = json.load( fp)
    
#print(temp["AR6ZS5N1187FB4595A"])

import pickle
filehandler = open("dataset.txt","rb")
dataset = pickle.load(filehandler)
for instance in dataset:
    if str(instance['id']) == "TRAACER128F4290F96":
        print('here')