"""
Turning the hyperparameter(alpha) of LDA

"""
#%% 1. read data
import pandas as pd
# if useParagraphs == True, split address into separate paragraphs
useParagraphs = True;
if useParagraphs == False:
    data = pd.read_csv('../data/state_of_the_union_corpus_full.csv')
else:
    data = pd.read_csv('../data/state_of_the_union_corpus_para.csv')

#%% 1.1 Choose Presidents
# prsdList = ['Bush1','Cliton','Bush2','Obama','Trump']
#prsdList = ['Bush1','Trump']
#data = data[[(prsd in prsdList) for prsd in data['president']]]

#%% 2.1 Split
from sklearn.cross_validation import train_test_split
test_size = 0.9
dataTrain, dataTest = train_test_split(data, random_state=1, test_size=test_size)
trainY = dataTrain['president']
testY = dataTest['president']    

#%% 3 Compute tf and tfidf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                ngram_range=(1, 2),
                                stop_words='english')
                                #max_features=850)
tf_train = tf_vectorizer.fit_transform(dataTrain['script'])
tf_test = tf_vectorizer.transform(dataTest['script'])

# Word TFiDF using scikit learn
tfidf_vectorizer = TfidfVectorizer(analyzer='word',
                       #stop_words='english',
                       #max_features = 850, 
                       ngram_range=(1, 2),
                       binary=False, lowercase=True, norm=None, smooth_idf=True, strip_accents=None,
                       sublinear_tf=True, use_idf=False)

tfidf_train = tfidf_vectorizer.fit_transform(dataTrain['script'])
tfidf_test = tfidf_vectorizer.transform(dataTest['script'])


#%% 4. Traing LDA Model
# Set Parameters
dim = 15
alphaList = [0.000001, 0.01, 0.1, 1, 5, 10, 20, 1000]
for alpha in alphaList:
    print('ALPHA: {}'.format(alpha))
    from sklearn.decomposition import LatentDirichletAllocation
    lda = LatentDirichletAllocation(n_components=dim, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    doc_topic_prior = alpha,
                                    random_state=0)
    lda.fit(tf_train)
    trainTopicDistArr = lda.transform(tf_train)
    testTopicDistArr = lda.transform(tf_test)
    
    #%% 5. Classification using Naive Bayes
    # Train Model
    from sklearn.naive_bayes import MultinomialNB
    nb_LDA = MultinomialNB().fit(trainTopicDistArr, trainY)
    
    # Print Training Accuracy
    from sklearn.cross_validation import cross_val_score
    print ("(CV, LDA): ", cross_val_score(MultinomialNB(), trainTopicDistArr,trainY, cv=3, scoring="accuracy").mean())
    
    # Print Test Accuracy
    print('(TE, LDA): '+str(nb_LDA.score(testTopicDistArr, testY)))