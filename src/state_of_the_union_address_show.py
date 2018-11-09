"""
Show the topics learned by LDA with different alpha values
"""
#%% 1. Read data
import pandas as pd

#import os, re
#txtPath = './demo_document/'
#
#df = pd.DataFrame(columns = ['script']);
#for filename in os.listdir(txtPath):                
#    f = open(txtPath+filename, 'r')
#    content = f.read()
#    f.close()
#    
#    df = df.append({'script':content},ignore_index=True)
    
# df.to_csv('./demo_document.csv',index = False)


data = pd.read_csv('./demo_document.csv')
#X = data['normalized_text']
X = data['script']

#%% 1.2 Processing: lower, normalize, remove stop words
# Set Stop Words
#from nltk.corpus import stopwords
#stop_words = stopwords.words('english')
#swFile = open("stop_words.txt")
#swFromFile = swFile.read().split()
#swFile.close();     stop_words += swFromFile
##stop_words.update('','-','--',)
#stop_words.append('')
#stop_words.append('-')
#stop_words.append('--')
#stop_words.append('thats')
#stop_words.append('thing')
#stop_words.append('weve')
##stop_words.append('')
#
## https://stackoverflow.com/questions/33587667/extracting-all-nouns-from-a-text-file-using-nltk/33608615
#is_noun = lambda pos: pos[:2] == 'NN'
#
#
#import re
#import nltk
#lem = nltk.stem.wordnet.WordNetLemmatizer()
#
##for scrIdx, row in data_raw.iterrows():
#for scriptIdx, script in X.iteritems():
#    # read raw string and lowering
#    strWordlist = script.lower().split()
#    
#    # remove all punctuations from string word
#    strWordlist = [re.sub(r'[^\w\s]','',word) for word in strWordlist]
#    
#    # Remove stopWords and normalize as noun
#    strWordlist = [lem.lemmatize(word,'n') for word in strWordlist if word not in stop_words]
#    
#    # Remain onyl nouns
#    strWordlist = [word for (word, pos) in nltk.pos_tag(strWordlist) if is_noun(pos)] 
#    
#    X.at[scriptIdx] = ' '.join(strWordlist)
#    # Save preprocessed data
#    # data = data.append({'year':row['year'],'president':row['president'], 'script':' '.join(strWordlist)},ignore_index=True)
    
    

#%% 2. TF
from sklearn.feature_extraction.text import CountVectorizer
tf_vectorizer = CountVectorizer(#max_df=0.9, #min_df=2,
                                        ngram_range=(1, 2),
                                        stop_words='english')
tf = tf_vectorizer.fit_transform(X)

#%% 3. LDA
topic_num = 3
alpha = 1e-4
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=topic_num, max_iter=5,
                                doc_topic_prior = alpha,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)
trainTopicDistArr = lda.transform(tf)

#%% 4 Show Topics
# https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
#def display_topics(model, feature_names, no_top_words):
#    for topic_idx, topic in enumerate(model.components_):
#        print("Topic %d:" % (topic_idx))
#        print(" ".join([feature_names[i]
#                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
#
#no_top_words = 20
#tf_feature_names = tf_vectorizer.get_feature_names()
#display_topics(lda, tf_feature_names, no_top_words)

#%% 5. Show topic words
# http://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
n_top_words = 10
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)