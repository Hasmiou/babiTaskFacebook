from __future__ import print_function
from functools import reduce
import re
import tarfile

import numpy as np

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

def tokenize(sent):
    '''Renvoie les jetons d'une phrase, y compris la ponctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def parse_stories(lines, only_supporting=False):
    '''Parseur des Histoires fournies dans le format de bAbi tasks
    Si only_supporting est true,
    seules les phrases qui soutiennent la réponse sont conservées.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1) #Identifiant + la phrase (ligne)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t') #Question + answer
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories : trouver ttes les sous histoires
                substory = [x for x in story if x] 
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data

def get_stories(f, only_supporting=False, max_length=None):
    '''
    Étant donné un nom de fichier, lisez le fichier, récupérez les histoires,
    puis convertissez les phrases en une seule histoire.
    Si max_length est fourni,
    les histoires plus longues que les jetons max_length seront ignorées.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data 
            if not max_length or len(flatten(story)) < max_length]
    return data


try:
    path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Erreur de téléchargement, Veillez le télécharger manuellement'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise


tar = tarfile.open(path)
challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'

train = get_stories(tar.extractfile(challenge.format('train')))
#print(train)

test = get_stories(tar.extractfile(challenge.format('test')))
#print(test)

vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train + test)))
#print('Taille initiale du vocabulaire', len(vocab))
#print(vocab)

def unique(l):
    ret = set()
    for el in l:
        ret.add(el)
    return list(ret)

flatten = lambda data: reduce(lambda x, y: x + y, data)

caracteres = unique(flatten(map(list,vocab)))
#print(caracteres)

sentences_from_stories = list(map(lambda d : d[0],train+test))
#print(sentences_from_stories)

distribution = dict()
for s in sentences_from_stories:
    len_s = len(s)
    if len_s in distribution.keys():
        distribution[len_s] +=1
    else:
        distribution[len_s] = 1
        
for i in range(100):
    if i not in distribution.keys():
        distribution[i]=0

"""
print("Taille du vocabulaire: {}".format(len(vocab)))
print("Nombre de c aractères: {}".format(len(caracteres)))
print("Nombre de stories (apprentissage) : {}".format(len(test)))
print("Nombre de stories (test) : {}".format(len(test)))
"""
