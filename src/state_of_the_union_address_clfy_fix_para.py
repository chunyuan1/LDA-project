"""
Use LDA as documentation vectorization method
and cLassify 

"""

#%% 1. read data
import pandas as pd
# if useParagraphs == True, split address into separate paragraphs
useParagraphs = True;
if useParagraphs == False:
    data = pd.read_csv('../data/state_of_the_union_corpus_full.csv')
else:
    data = pd.read_csv('../data/state_of_the_union_corpus_para.csv')

#%% 1.1 Set Parameters
dimList = [5, 10, 20, 50]    
alphaList = [1e-4, 1e-2, 1, 10, 50]
testSizeList = [0.1, 0.3, 0.5, 0.8, 0.9]
usingTfiDFList = [0, 1]

# dim = 10
print('Dim: {}'.format(dim))
#%% 1.1 Choose Presidents
# prsdList = ['Bush1','Cliton','Bush2','Obama','Trump']
prsdList = ['Bush1','Trump']
data = data[[(prsd in prsdList) for prsd in data['president']]]

#%% 2.1 Split
from sklearn.cross_validation import train_test_split
test_size = 0.7
print('Test Size: {}'.format(test_size))
dataTrain, dataTest = train_test_split(data, random_state=1, test_size=test_size)
trainY = dataTrain['president']
testY = dataTest['president']    

#%% 3. Compute tf and tfidf
print('Computing tf and tfidf')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                ngram_range=(1, 2),
                                stop_words='english')
tf_vectorizer_dim = CountVectorizer(max_df=0.95, min_df=2,
                                ngram_range=(1, 2),
                                stop_words='english',
                                max_features=dim)

tf_train = tf_vectorizer.fit_transform(dataTrain['script'])
tf_test = tf_vectorizer.transform(dataTest['script'])
tf_train_dim = tf_vectorizer_dim.fit_transform(dataTrain['script'])
tf_test_dim = tf_vectorizer_dim.transform(dataTest['script'])

# Word TFiDF using scikit learn
tfidf_vectorizer = TfidfVectorizer(analyzer='word',
                       stop_words='english',
                       ngram_range=(1, 2),
                       binary=False, lowercase=True, norm=None, smooth_idf=True, strip_accents=None,
                       sublinear_tf=True, use_idf=False)
tfidf_vectorizer_dim = TfidfVectorizer(analyzer='word',
                       stop_words='english',
                       max_features = dim, 
                       ngram_range=(1, 2),
                       binary=False, lowercase=True, norm=None, smooth_idf=True, strip_accents=None,
                       sublinear_tf=True, use_idf=False)

tfidf_train = tfidf_vectorizer.fit_transform(dataTrain['script'])
tfidf_test = tfidf_vectorizer.transform(dataTest['script'])
tfidf_train_dim = tfidf_vectorizer_dim.fit_transform(dataTrain['script'])
tfidf_test_dim = tfidf_vectorizer_dim.transform(dataTest['script'])

#%% 4. Traing LDA Model
alpha = 1e-4
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=dim, max_iter=5,
                                doc_topic_prior = alpha,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
usingTFiDF = 0
if usingTFiDF:
    print('Traing LDA Model(TFidf), Alpha: {}'.format(alpha))
    lda.fit(tfidf_train)
    trainTopicDistArr = lda.transform(tfidf_train)
    testTopicDistArr = lda.transform(tfidf_test)
else:
    print('Traing LDA Model(TF), Alpha: {}'.format(alpha))
    lda.fit(tf_train)
    trainTopicDistArr = lda.transform(tf_train)
    testTopicDistArr = lda.transform(tf_test)
#%% 4.1 Show Topics
# https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
tf_feature_names = tf_vectorizer.get_feature_names()
# display_topics(lda, tf_feature_names, no_top_words)

#%% 5. NMF and Classification using Naive Bayes
print('NMF and Classification using Naive Bayes')
# NMF(SLOW!)
from sklearn.decomposition import NMF
nmfModel = NMF(n_components=dim, init='random', random_state=0)
tfidf_train_nmf = nmfModel.fit_transform(tfidf_train)
tfidf_test_nmf = nmfModel.fit_transform(tfidf_test)
tf_train_nmf = nmfModel.fit_transform(tf_train)
tf_test_nmf = nmfModel.fit_transform(tf_test)

# Train Models
from sklearn.naive_bayes import MultinomialNB
nb_LDA = MultinomialNB().fit(trainTopicDistArr, trainY)
nb_TF = MultinomialNB().fit(tf_train, trainY)
nb_TFiDF = MultinomialNB().fit(tfidf_train, trainY)
nb_TF_dim = MultinomialNB().fit(tf_train_dim, trainY)
nb_TFiDF_dim = MultinomialNB().fit(tfidf_train_dim, trainY)
nb_TF_nmf = MultinomialNB().fit(tf_train_nmf, trainY)
nb_TFiDF_nmf = MultinomialNB().fit(tfidf_train_nmf, trainY)

# Print Training Accuracy
from sklearn.cross_validation import cross_val_score
print ("(CV, TF):\t\t", cross_val_score(MultinomialNB(), tf_train,trainY, cv=3, scoring="accuracy").mean())
print ("(CV, TFIDF):\t\t", cross_val_score(MultinomialNB(), tfidf_train,trainY, cv=3, scoring="accuracy").mean())
print ("(CV, TF+DIM):\t\t", cross_val_score(MultinomialNB(), tf_train_dim,trainY, cv=3, scoring="accuracy").mean())
print ("(CV, TFIDF+DIM):\t", cross_val_score(MultinomialNB(), tfidf_train_dim,trainY, cv=3, scoring="accuracy").mean())
print ("(CV, TF+NMF):\t\t", cross_val_score(MultinomialNB(), tf_train_nmf,trainY, cv=3, scoring="accuracy").mean())
print ("(CV, TFIDF+NMF):\t", cross_val_score(MultinomialNB(), tfidf_train_nmf,trainY, cv=3, scoring="accuracy").mean())
print ("(CV, LDA):\t\t", cross_val_score(MultinomialNB(), trainTopicDistArr,trainY, cv=3, scoring="accuracy").mean())

# Print Test Accuracy
print('(TE, TF):\t\t'+str(nb_TF.score(tf_test, testY)))
print('(TE, TFIDF):\t\t'+str(nb_TFiDF.score(tfidf_test, testY)))
print('(TE, TF+DIM):\t\t'+str(nb_TF_dim.score(tf_test_dim, testY)))
print('(TE, TFIDF+DIM): \t'+str(nb_TFiDF_dim.score(tfidf_test_dim, testY)))
print('(TE, TF+NMF):\t\t'+str(nb_TF_nmf.score(tf_test_nmf, testY)))
print('(TE, TFIDF+NMF):\t'+str(nb_TFiDF_nmf.score(tfidf_test_nmf, testY)))
print('(TE, LDA):\t\t'+str(nb_LDA.score(testTopicDistArr, testY)))

#%% 6. LSA and Classification using SVC
#print('LSA and Classification using SVC')
## SVD
#from sklearn.decomposition import TruncatedSVD
#from sklearn.preprocessing import Normalizer
#from sklearn.pipeline import make_pipeline
#svd = TruncatedSVD(dim)
#normalizer = Normalizer(copy=False)
#lsa = make_pipeline(svd, normalizer)
#tfidf_train_svd = lsa.fit_transform(tfidf_train)
#tfidf_test_svd = lsa.fit_transform(tfidf_test)
#tf_train_svd = lsa.fit_transform(tf_train)
#tf_test_svd = lsa.fit_transform(tf_test)
#
## Training Models
#from sklearn.svm import SVC
#svc_LDA = SVC().fit(trainTopicDistArr, trainY)
#svc_TF = SVC().fit(tf_train, trainY)
#svc_TFIDF = SVC().fit(tfidf_train, trainY)
#svc_TF_dim = SVC().fit(tf_train_dim, trainY)
#svc_TFIDF_dim = SVC().fit(tfidf_train_dim, trainY)
#svc_TF_svd = SVC().fit(tf_train_svd, trainY)
#svc_TFIDF_svd = SVC().fit(tfidf_train_svd, trainY)
#
## Print Training Accuacy
#from sklearn.cross_validation import cross_val_score
#print ("(CV, TF):\t\t", cross_val_score(SVC(), tf_train,trainY, cv=3, scoring="accuracy").mean())
#print ("(CV, TFIDF):\t\t", cross_val_score(SVC(), tfidf_train,trainY, cv=3, scoring="accuracy").mean())
#print ("(CV, TF+DIM):\t\t", cross_val_score(SVC(), tf_train_dim,trainY, cv=3, scoring="accuracy").mean())
#print ("(CV, TFIDF+DIM):\t", cross_val_score(SVC(), tfidf_train_dim,trainY, cv=3, scoring="accuracy").mean())
#print ("(CV, TF+SVD):\t\t", cross_val_score(SVC(), tf_train_svd,trainY, cv=3, scoring="accuracy").mean())
#print ("(CV, TFIDF+SVD):\t", cross_val_score(SVC(), tfidf_train_svd,trainY, cv=3, scoring="accuracy").mean())
#print ("(CV, LDA):\t\t", cross_val_score(SVC(), trainTopicDistArr,trainY, cv=3, scoring="accuracy").mean())
#
## Print Testing Accuacy
#print('(TE, TF):\t\t'+str(svc_TF.score(tf_test, testY)))
#print('(TE, TFIDF):\t\t'+str(svc_TFIDF.score(tfidf_test, testY)))
#print('(TE, TF+DIM):\t\t'+str(svc_TF_dim.score(tf_test_dim, testY)))
#print('(TE, TFIDF+DIM):\t'+str(svc_TFIDF_dim.score(tfidf_test_dim, testY)))
#print('(TE, TF+SVD):\t\t'+str(svc_TF_svd.score(tf_test_svd, testY)))
#print('(TE, TFIDF+SVD):\t'+str(svc_TFIDF_svd.score(tfidf_test_svd, testY)))
#print('(TE, LDA):\t\t'+str(svc_LDA.score(testTopicDistArr, testY)))