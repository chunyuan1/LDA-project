"""
- Do same things on different dataset:
1. non-bayesian method(MLE)
2. Bayesian method with uniform prior
3. Bayesian method with prior learn form full data/domain knowledge

- Dataset
- Bayesian/nonBayesian method
- Large/small dataset
- uniform/learned prior
- (unsuccessful attempt on learn speaker directly)

1. State of the Union Corpus (1989 - 2017)
- This should be a very easy task. Use this to make sure libraries are correctly used
- Also observe if topics changed with time
- https://www.kaggle.com/rtatman/state-of-the-union-corpus-1989-2017/data
"""
#%% 0. Preprocessing
"""
import pandas as pd
import os, re
txtPath = '../data/state/'

def transformName(name,year):
    if name=='Bush':
        if year<=1992:
            return 'Bush1'
        else:
            return 'Bush2'
    return name

df = pd.DataFrame(columns = ['year','president','script']);
for filename in os.listdir(txtPath):        
    year = int(re.findall(string=filename,pattern='\d{4}')[0])
    presidentName = transformName(re.findall(string=filename,pattern='^(.+?)_')[0],year)
    
    f = open(txtPath+filename, 'r', encoding='utf-8')
    content = f.read()
    f.close()
    
    df = df.append({'year':year,'president':presidentName,'script':content},ignore_index=True)
    
df.to_csv('../data/state_of_the_union_corpus.csv',index = False)
"""

#%% 1. read data
import pandas as pd
data_raw = pd.read_csv('../data/state_of_the_union_corpus.csv')

# if useParagraphs == True, split address into separate paragraphs
useParagraphs = False;

#%% 2. Processing: lower, normalize, remove stop words
# Set Stop Words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
swFile = open("stop_words.txt")
swFromFile = swFile.read().split()
swFile.close();     stop_words += swFromFile
#stop_words.update('','-','--',)
stop_words.append('')
stop_words.append('-')
stop_words.append('--')
#stop_words.append('')

import re
import nltk
lem = nltk.stem.wordnet.WordNetLemmatizer()
if useParagraphs == False:
    data = pd.DataFrame(columns = data_raw.columns)
    for scrIdx, row in data_raw.iterrows():
        # read raw string and lowering
        strWordlist = row['script'].lower().split()
        
        # remove all punctuations from string word
        strWordlist = [re.sub(r'[^\w\s]','',word) for word in strWordlist]
        
        # Remove stopWords and normalize as noun
        strWordlist = [lem.lemmatize(word,'n') for word in strWordlist if word not in stop_words]
        
        # Remain onyl nouns
        # strMain = [word for (word, pos) in nltk.pos_tag(strMain) if is_noun(pos)] 
        
        # Save preprocessed data
        data = data.append({'year':row['year'],'president':row['president'], 'script':' '.join(strWordlist)},ignore_index=True)
else:
    paraLenThres = 20
    data = pd.DataFrame(columns = data_raw.columns)
    for textIdx, row in data_raw.iterrows():
        textParaList = row['script'].lower().split('\n')        
        for paraIdx, para in enumerate(textParaList):
            if len(para) < paraLenThres:
                continue
            
            # lowering
            paraWordlist = para.lower().split()
        
            # remove all punctuations from string word
            paraWordlist = [re.sub(r'[^\w\s]','',word) for word in paraWordlist]
            
            # Remove stopWords and normalize as noun
            paraWordlist = [lem.lemmatize(word,'n') for word in paraWordlist if word not in stop_words]
            
            # Remain onyl nouns
            # strMain = [word for (word, pos) in nltk.pos_tag(strMain) if is_noun(pos)] 
            
            # Save preprocessed data
            if len(paraWordlist) >= paraLenThres:
                data = data.append({'year':row['year'],'president':row['president'], 'script':' '.join(paraWordlist)},ignore_index=True)

#%% 3 Build Dictionary and training corpus (including all possible words, therefore should use the full dataset to avoid "undefined word")
scripts = list(data['script'])
presidents = list(data['president'])
scriptsList = [script.split() for script in scripts]

from gensim import corpora,models
# Input of Dictionary(): list of list of words, e.g. [['I','have','a','pen'],['I','have','an','apple']]
dictionary = corpora.Dictionary(scriptsList)

# Compute Word Frequency (for words in train data)
corpus = [dictionary.doc2bow(script) for script in scriptsList]

tfidf_model = models.TfidfModel(corpus) 
corpus_tfidf = tfidf_model[corpus]

#%% 4. Traing LDA Model
from gensim.models.ldamodel import LdaModel
num_topics = 150
lda = LdaModel(corpus=corpus_tfidf,id2word=dictionary,num_topics=num_topics)
# lda.print_topics()

#%% 5.1 Predict(One piece)
#obamaScript1 = data[data['president']=='Obama']['script'].iloc[0]
#obamaCorpus1 = dictionary.doc2bow(obamaScript1.split())
#
## Vectorization and predict
#obamaScriptsTFiDF = tfidf_model[obamaCorpus1]
#
#lda.inference([obamaScriptsTFiDF])[0]

#%% 5.2 Predict(One President)
#obamaScript = data[data['president']=='Obama']['script']
#obamaCorpus = [dictionary.doc2bow(script.split()) for script in obamaScript]
#obamaScriptsTFiDF = tfidf_model[obamaCorpus]
#lda.inference(obamaScriptsTFiDF)[0]
#
#trumpScript = data[data['president']=='Trump']['script']
#trumpCorpus = [dictionary.doc2bow(script.split()) for script in trumpScript]
#trumpScriptsTFiDF = tfidf_model[trumpCorpus]
#lda.inference(trumpScriptsTFiDF)[0]

#%% 5.3 Predict(Another President)
#clintonScript = data[data['president']=='Clinton']['script']
#clintonCorpus = [dictionary.doc2bow(script.split()) for script in clintonScript]
#clintonScriptsTFiDF = tfidf_model[clintonCorpus]
#lda.inference(clintonScriptsTFiDF)[0]
#
#bush1Script = data[data['president']=='Bush1']['script']
#bush1Corpus = [dictionary.doc2bow(script.split()) for script in bush1Script]
#bush1ScriptsTFiDF = tfidf_model[bush1Corpus]
#lda.inference(bush1ScriptsTFiDF)[0]

#%% 5.4 Get Prototype for each president
from collections import Counter
import numpy as np
prsdCounter = Counter(presidents)
topicDistRecord = np.zeros(shape=(num_topics,len(prsdCounter)))   # Each column is one account's topic distribution
for prsdIdx, prsdName in enumerate(prsdCounter.keys()):
    prsdScripts = list(data[data['president']==prsdName]['script'])
    if len(prsdScripts) > 1:
        prsdScript = ' '.join(prsdScripts)
    else:
        prsdScript = prsdScripts[0]
    prsdCorpus1 = dictionary.doc2bow(prsdScript.split())
    prsdScriptsTFiDF = tfidf_model[prsdCorpus1]
    
    topicDistRecord[:,prsdIdx] = lda.inference([prsdScriptsTFiDF])[0]
    # Normalize
    topicDistRecord[:,prsdIdx] = topicDistRecord[:,prsdIdx]/np.sum(topicDistRecord[:,prsdIdx])

# Visualize topic distribution of each president
#import matplotlib.pyplot as plt
#plt.plot(topicDistRecord)
#plt.legend(list(prsdCounter.keys()))
    
#%% 5.5 Predict all scripts
def findAccByDist(topicDistArr, topicDistRecord):
    topicDistRow = np.matrix(topicDistArr)
    # topicDistRow = topicDistRow/np.sum(topicDistRow,0)
    topicDistRecordMat = np.matrix(topicDistRecord)
    return [int(maxIdx) for maxIdx in np.argmax(topicDistRow*topicDistRecordMat,1)]


topicDistArr = lda.inference(corpus_tfidf)[0]
allPrsdPredID = findAccByDist(topicDistArr, topicDistRecord)

#%% 6.1 Label Alignment for LDA
mapDictFromTrueToCluID = dict()
mapDictFromCluToTrueID = dict()
import collections
for prsdIdx in range(len(prsdCounter)):
    # Speaker true ID and corresponding scripts
    prsdTrueName = list(prsdCounter.keys())[prsdIdx]
    prsdScripts = list(data[data['president']==prsdTrueName]['script'])
    
    # Vectorization and predict
    prsdCorpus = [dictionary.doc2bow(script.split()) for script in prsdScripts]
    prsdCorpus_tfidf = tfidf_model[prsdCorpus]
    prsdTopicDistArr = lda.inference(prsdCorpus_tfidf)[0]    
    prsdPredIDArr = findAccByDist(prsdTopicDistArr, topicDistRecord)
    
    # Find majority prediction
    counter = collections.Counter(prsdPredIDArr)
    prsdPredID = str(counter.most_common(1)[0][0])
    
    # Update mapping dict
    mapDictFromTrueToCluID[prsdTrueName] = prsdPredID
    mapDictFromCluToTrueID[prsdPredID] = prsdTrueName

#%% 6.2 Accuracy(LDA)
#predictSpeakerScriptsTF = tf_vectorizer.transform(X)
#predictSpeakers = np.argmax(lda.transform(predictSpeakerScriptsTF),axis=1)
correctCount = 0
# ylist = list(y)
for textIdx in range(len(data)):
    prsdTrueName = data.iloc[textIdx]['president']
    prsdPredIDAligned = mapDictFromTrueToCluID[prsdTrueName]
    if prsdPredIDAligned == str(allPrsdPredID[textIdx]):
        correctCount += 1
acc = correctCount/len(data)
print('ACC of LDA: ', str(acc))    

#%% 7. K-Means
# Reference: Clustering text documents using k-means
# > scikit-learn.org/stable/auto_examples/text/document_clustering.html
# Vectorize using scikit learn
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(analyzer='word',
                       #stop_words='english',
                       #max_features = 850, 
                       ngram_range=(1, 2),
                       binary=False, lowercase=True, norm=None, smooth_idf=True, strip_accents=None,
                       sublinear_tf=True, use_idf=False)

scikit_tfidf = tfidf_vectorizer.fit_transform(data['script'])

from sklearn.cluster import KMeans
true_k = 5
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
km.fit(scikit_tfidf)
kmeansPred = km.labels_