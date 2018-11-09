"""
Gender classification using blog data

"""
#%% 1. Read Data
import numpy as np
import pandas as pd
blogRaw = pd.read_csv('../data/blog-gender-dataset-proc.csv')

#%%
dim=30

#%% 2.1 Split
from sklearn.cross_validation import train_test_split
dataTrain, dataTest = train_test_split(blogRaw, random_state=1, test_size=0.5)

    
#%% 3 Build Dictionary and training corpus (including all possible words, therefore should use the full dataset to avoid "undefined word")
#trainScripts = list(dataTrain['blog'])
#testScripts = list(dataTest['blog'])
#trainPresidents = list(dataTrain['gender'])
#trainScriptsList = [script.split() for script in trainScripts]
#testScriptsList = [script.split() for script in testScripts]
#
#from gensim import corpora,models
## Input of Dictionary(): list of list of words, e.g. [['I','have','a','pen'],['I','have','an','apple']]
#dictionary = corpora.Dictionary(trainScriptsList)
#
## Compute Word Frequency (for words in train data)
#trainCorpus = [dictionary.doc2bow(script) for script in trainScriptsList]
#testCorpus = [dictionary.doc2bow(script) for script in testScriptsList]
#
#train_tfidf_model = models.TfidfModel(trainCorpus)
#train_corpus_tfidf = train_tfidf_model[trainCorpus]
#test_corpus_tfidf = train_tfidf_model[testCorpus]

# Word TF using scikit learn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                ngram_range=(1, 2),
                                stop_words='english',
                                #max_features=dim
                                )
sklearn_tf_train = tf_vectorizer.fit_transform(dataTrain['blog'])
sklearn_tf_test = tf_vectorizer.transform(dataTest['blog'])

# Word TFiDF using scikit learn
tfidf_vectorizer = TfidfVectorizer(analyzer='word',
                       stop_words='english',
                       # max_features = dim, 
                       ngram_range=(1, 2),
                       binary=False, lowercase=True, norm=None, smooth_idf=True, strip_accents=None,
                       sublinear_tf=True, use_idf=False)

sklearn_tfidf_train = tfidf_vectorizer.fit_transform(dataTrain['blog'])
sklearn_tfidf_test = tfidf_vectorizer.transform(dataTest['blog'])

# LSA
# http://scikit-learn.org/stable/auto_examples/text/document_clustering.html#example-text-document-clustering-py
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
svd = TruncatedSVD(dim)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
sklearn_tfidf_train_svd = lsa.fit_transform(sklearn_tfidf_train)
sklearn_tfidf_test_svd = lsa.fit_transform(sklearn_tfidf_test)
sklearn_tf_train_svd = lsa.fit_transform(sklearn_tf_train)
sklearn_tf_test_svd = lsa.fit_transform(sklearn_tf_test)



from sklearn.decomposition import NMF
nmfModel = NMF(n_components=dim, init='random', random_state=0)
sklearn_tfidf_train_nmf = nmfModel.fit_transform(sklearn_tfidf_train)
sklearn_tfidf_test_nmf = nmfModel.fit_transform(sklearn_tfidf_test)
sklearn_tf_train_nmf = nmfModel.fit_transform(sklearn_tf_train)
sklearn_tf_test_nmf = nmfModel.fit_transform(sklearn_tf_test)

#sklearn_tf_train_nmf = nmfModel.fit_transform(sklearn_tfidf_train)



#%% 4. Traing LDA Model and Vectoring
# Train model
#from gensim.models.ldamodel import LdaModel
#lda = LdaModel(corpus=train_corpus_tfidf, id2word=dictionary, num_topics=dim)#, alpha=alpha)
#
## Vectorizing
#trainTopicDistArr = lda.inference(train_corpus_tfidf)[0]
#testTopicDistArr = lda.inference(test_corpus_tfidf)[0]

# Train model
from sklearn.decomposition import NMF, LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=dim, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(sklearn_tfidf_train)
trainTopicDistArr = lda.transform(sklearn_tfidf_train)
testTopicDistArr = lda.transform(sklearn_tfidf_test)


#%% 5. Classification using Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score
#nb_SP_Model = MultinomialNB()
#nb_SP_Model.fit(sklearn_tfidf, gender_data_train['gender'])
nb_LDA = MultinomialNB().fit(trainTopicDistArr, dataTrain['gender'])
nb_TF = MultinomialNB().fit(sklearn_tf_train_nmf, dataTrain['gender'])
nb_TFiDF = MultinomialNB().fit(sklearn_tfidf_train_nmf, dataTrain['gender'])
# print ("Model accuracy within dataset: ", nb_SP_Model.score(sklearn_tfidf, gender_data_train['gender']))

print ("(TF)Model accuracy within dataset: ", nb_TF.score(sklearn_tf_train_nmf, dataTrain['gender']))
print ("(TFIDF)Model accuracy within dataset: ", nb_TFiDF.score(sklearn_tfidf_train_nmf, dataTrain['gender']))
print ("(LDA)Model accuracy within dataset: ", nb_LDA.score(trainTopicDistArr, dataTrain['gender']))

#print ("Model accuracy with cross validation:", cross_val_score(MultinomialNB(), sklearn_tfidf, 
#                                                                dataTrain['president'], cv=3, scoring="accuracy").mean())
print('TF: '+str(nb_TF.score(sklearn_tf_test_nmf, dataTest['gender'])))
print('TFIDF: '+str(nb_TFiDF.score(sklearn_tfidf_test_nmf, dataTest['gender'])))
print('LDA: '+str(nb_LDA.score(testTopicDistArr, dataTest['gender'])))

#%% 6. Classification using SVC
#from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
# from sklearn.cross_validation import cross_val_score
trainY = dataTrain['gender']
testY = dataTest['gender']
svc_LDA = SVC().fit(trainTopicDistArr, trainY)
svc_TF = SVC().fit(sklearn_tf_train_svd, trainY)
svc_TFIDF = SVC().fit(sklearn_tfidf_train_svd, trainY)

print ("(TF)Model accuracy within dataset: ", svc_TF.score(sklearn_tf_train_svd, testY))
print ("(TFIDF)Model accuracy within dataset: ", svc_TFIDF.score(sklearn_tfidf_train_svd, testY))
print ("(LDA)Model accuracy within dataset: ", svc_LDA.score(trainTopicDistArr, testY))

print('TF: '+str(svc_TF.score(sklearn_tf_test_svd, testY)))
print('TFIDF: '+str(svc_TFIDF.score(sklearn_tfidf_test_svd, testY)))
print('LDA: '+str(svc_LDA.score(testTopicDistArr, testY)))