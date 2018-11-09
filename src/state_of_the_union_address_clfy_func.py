"""
Use LDA as documentation vectorization method
and cLassify 

"""
def state_of_the_union_address_clfy_func(data, test_size, dim, alpha, usingTFiDF, cvFoldNum=5, rerunNum=5, verbose = 1):
    cvRecord = {'tf':0, 'tfidf':0, 'tf_dim':0, 'tfidf_dim':0, 'tf_nmf':0, 'tfidf_nmf':0, 'LDA':0}
    testRecord = {'tf':0, 'tfidf':0, 'tf_dim':0, 'tfidf_dim':0, 'tf_nmf':0, 'tfidf_nmf':0, 'LDA':0}
    for rerunIdx in range(rerunNum):
        if verbose: print('{}/{} re-run: '.format(rerunIdx+1, rerunNum))
        #%% 2.1 Split
        from sklearn.cross_validation import train_test_split
        if verbose: print('Dim: {}'.format(dim))
        if verbose: print('Test Size: {}'.format(test_size))
        dataTrain, dataTest = train_test_split(data, random_state=1, test_size=test_size)
        trainX = dataTrain['script']
        trainY = dataTrain['president']
        testX = dataTest['script']
        testY = dataTest['president']        
    
    
        #%% 3. Compute tf and tfidf
        if verbose: print('Computing tf and tfidf')
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        ngram_range=(1, 2),
                                        stop_words='english')
        tf_vectorizer_dim = CountVectorizer(max_df=0.95, min_df=2,
                                        ngram_range=(1, 2),
                                        stop_words='english',
                                        max_features=dim)
        
        tf_train = tf_vectorizer.fit_transform(trainX)
        tf_test = tf_vectorizer.transform(testX)
        tf_train_dim = tf_vectorizer_dim.fit_transform(trainX)
        tf_test_dim = tf_vectorizer_dim.transform(testX)
        
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
        
        tfidf_train = tfidf_vectorizer.fit_transform(trainX)
        tfidf_test = tfidf_vectorizer.transform(testX)
        tfidf_train_dim = tfidf_vectorizer_dim.fit_transform(trainX)
        tfidf_test_dim = tfidf_vectorizer_dim.transform(testX)
    
        #%% 4. Traing LDA Model
        from sklearn.decomposition import LatentDirichletAllocation
        lda = LatentDirichletAllocation(n_components=dim, max_iter=5,
                                        doc_topic_prior = alpha,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        if usingTFiDF:
            if verbose: print('Traing LDA Model(TFidf), Alpha: {}'.format(alpha))
            lda.fit(tfidf_train)
            trainTopicDistArr = lda.transform(tfidf_train)
            testTopicDistArr = lda.transform(tfidf_test)
        else:
            if verbose: print('Traing LDA Model(TF), Alpha: {}'.format(alpha))
            lda.fit(tf_train)
            trainTopicDistArr = lda.transform(tf_train)
            testTopicDistArr = lda.transform(tf_test)
            
        #%% 4.1 Show Topics
        # https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
    #    def display_topics(model, feature_names, no_top_words):
    #        for topic_idx, topic in enumerate(model.components_):
    #            print("Topic %d:" % (topic_idx))
    #            print(" ".join([feature_names[i]
    #                            for i in topic.argsort()[:-no_top_words - 1:-1]]))
    #    
    #    no_top_words = 10
    #    tf_feature_names = tf_vectorizer.get_feature_names()
        # display_topics(lda, tf_feature_names, no_top_words)
    
        #%% 5. NMF and Classification using Naive Bayes
        if verbose: print('NMF and Classification using Naive Bayes')
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
        cvRecord['tf'] += cross_val_score(MultinomialNB(),           tf_train,trainY, cv=cvFoldNum, scoring="accuracy").mean()
        cvRecord['tfidf'] += cross_val_score(MultinomialNB(),        tfidf_train,trainY, cv=cvFoldNum, scoring="accuracy").mean()
        cvRecord['tf_dim'] += cross_val_score(MultinomialNB(),       tf_train_dim,trainY, cv=cvFoldNum, scoring="accuracy").mean()
        cvRecord['tfidf_dim'] += cross_val_score(MultinomialNB(),    tfidf_train_dim,trainY, cv=cvFoldNum, scoring="accuracy").mean()
        cvRecord['tf_nmf'] += cross_val_score(MultinomialNB(),       tf_train_nmf,trainY, cv=cvFoldNum, scoring="accuracy").mean()
        cvRecord['tfidf_nmf'] += cross_val_score(MultinomialNB(),    tfidf_train_nmf,trainY, cv=cvFoldNum, scoring="accuracy").mean()
        cvRecord['LDA'] += cross_val_score(MultinomialNB(),          trainTopicDistArr,trainY, cv=cvFoldNum, scoring="accuracy").mean()
        
        testRecord['tf'] +=         nb_TF.score(tf_test, testY)
        testRecord['tfidf'] +=      nb_TFiDF.score(tfidf_test, testY)
        testRecord['tf_dim'] +=     nb_TF_dim.score(tf_test_dim, testY)
        testRecord['tfidf_dim'] +=  nb_TFiDF_dim.score(tfidf_test_dim, testY)
        testRecord['tf_nmf'] +=     nb_TF_nmf.score(tf_test_nmf, testY)
        testRecord['tfidf_nmf'] +=  nb_TFiDF_nmf.score(tfidf_test_nmf, testY)
        testRecord['LDA'] +=        nb_LDA.score(testTopicDistArr, testY)        
    if verbose:
        print ("(CV, TF):\t\t",         cvRecord['tf']/rerunNum)
        print ("(CV, TFIDF):\t\t",      cvRecord['tfidf']/rerunNum)
        print ("(CV, TF+DIM):\t\t",     cvRecord['tf_dim']/rerunNum)
        print ("(CV, TFIDF+DIM):\t",    cvRecord['tfidf_dim']/rerunNum)
        print ("(CV, TF+NMF):\t\t",     cvRecord['tf_nmf']/rerunNum)
        print ("(CV, TFIDF+NMF):\t",    cvRecord['tfidf_nmf']/rerunNum)
        print ("(CV, LDA):\t\t",        cvRecord['LDA']/rerunNum)
        
        # Print Test Accuracy
        print('(TE, TF):\t\t',          testRecord['tf']/rerunNum)
        print('(TE, TFIDF):\t\t',       testRecord['tfidf']/rerunNum)
        print('(TE, TF+DIM):\t\t',      testRecord['tf_dim']/rerunNum)
        print('(TE, TFIDF+DIM): \t',    testRecord['tfidf_dim']/rerunNum)
        print('(TE, TF+NMF):\t\t',      testRecord['tf_nmf']/rerunNum)
        print('(TE, TFIDF+NMF):\t',     testRecord['tfidf_nmf']/rerunNum)
        print('(TE, LDA):\t\t',         testRecord['LDA']/rerunNum)

    import pandas as pd
    record = pd.DataFrame(columns = ['dim', 'alpha', 'test_size', 'method', 'usingtfidf', 'CV_score', 'test_score'])
    for methodIdx in range(len(cvRecord)):
        method = list(cvRecord.keys())[methodIdx]
        record = record.append({'dim':dim, 'alpha':alpha, 'test_size':test_size, 'method':method, 'usingtfidf': usingTFiDF, 'CV_score': cvRecord[method]/rerunNum, 'test_score':testRecord[method]/rerunNum},ignore_index=True)
    
    return record
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