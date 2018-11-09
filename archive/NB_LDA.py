# www.kaggle.com/agaleana/predicting-south-park-dialogues/notebook
#%% 1. Import Libaraies
import numpy as np
import matplotlib as plt
import pandas as pd


from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import make_pipeline

#%% 2. Import Dataset
South_Park_raw = pd.read_csv('../data/simpsons_script_lines.csv',
                                    error_bad_lines=False,
                                    warn_bad_lines=False,
                                    low_memory=False)

scriptLenThres = 15
South_Park_raw = South_Park_raw[(South_Park_raw['speaking_line']=='true') & (South_Park_raw['word_count']>scriptLenThres)]

# Head and shape of dataset
showData = False
if showData:
    print(South_Park_raw.head())
    print(South_Park_raw.shape)
    print(South_Park_raw.describe())
del(showData)

#%% 3. Extract Top speakers
def extractTopSpeakers(scripts_id, commonCount = 30):
    import collections
    counter = collections.Counter(scripts_id)
    mostCommonCharId = counter.most_common(commonCount)    
    mostCommonCharId = pd.Series(index = [i[0] for i in mostCommonCharId], data = [i[1] for i in mostCommonCharId])
    
    return mostCommonCharId

top_speakers = extractTopSpeakers(South_Park_raw['character_id'], commonCount = 2)
top_speakers_freq = top_speakers.values / np.sum(top_speakers.values)

minSpeakerCount = np.min(top_speakers.values)
main_char_lines_batch = pd.DataFrame(columns=['character_id','normalized_text'])
for speakerIdx in range(len(top_speakers)):
    main_char_lines_fullCol = South_Park_raw.loc[South_Park_raw['character_id']==top_speakers.index[speakerIdx]]
    main_char_lines = main_char_lines_fullCol.iloc[:minSpeakerCount][['character_id','normalized_text']]
    main_char_lines_batch = pd.concat([main_char_lines_batch,main_char_lines])

for rowIdx in range(len(main_char_lines_batch)):
    main_char_lines_batch.iloc[rowIdx].normalized_text = str(main_char_lines_batch.iloc[rowIdx].normalized_text)
main_char_lines = main_char_lines_batch
#top_speakers = South_Park_raw.groupby(['character_id']).size().loc[South_Park_raw.groupby(['character_id']).size() > 500]

#print (top_speakers.sort_values(ascending=False))

# Extract corresponding script lines
#main_char_lines_fullCol = pd.DataFrame(South_Park_raw.loc[South_Park_raw['character_id'].isin(top_speakers.index.values)])
#main_char_lines = main_char_lines_fullCol[['character_id','normalized_text']]
# main_char_lines_double = pd.concat([main_char_lines,main_char_lines])


#for rowIdx in range(len(main_char_lines)):
#    main_char_lines.iloc[rowIdx].normalized_text = str(main_char_lines.iloc[rowIdx].normalized_text)


# main_char_lines = main_char_lines.reset_index(drop=True)


#%% 4. Define training and test set
from sklearn.model_selection import train_test_split
# define X and y
X = main_char_lines_batch.normalized_text
y = main_char_lines_batch.character_id

# split the new DataFrame into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#%% 5. Search for best parameters to use in model
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#from sklearn.naive_bayes import MultinomialNB
#
#pipe = make_pipeline(TfidfVectorizer(), MultinomialNB())
##pipe.steps
#ifgridSearch = False
#if ifgridSearch:
#    param_grid = {}
#    param_grid["tfidfvectorizer__max_features"] = [500, 1000, 15000]
#    param_grid["tfidfvectorizer__ngram_range"] = [(1,1), (1,2), (2,2)]
#    param_grid["tfidfvectorizer__lowercase"] = [True, False]
#    param_grid["tfidfvectorizer__stop_words"] = ["english", None]
#    param_grid["tfidfvectorizer__strip_accents"] = ["ascii", "unicode", None]
#    param_grid["tfidfvectorizer__analyzer"] = ["word", "char"]
#    param_grid["tfidfvectorizer__binary"] = [True, False]
#    param_grid["tfidfvectorizer__norm"] = ["l1", "l2", None]
#    param_grid["tfidfvectorizer__use_idf"] = [True, False]
#    param_grid["tfidfvectorizer__smooth_idf"] = [True, False]
#    param_grid["tfidfvectorizer__sublinear_tf"] = [True, False]
#    
#    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
#    
#    # Helpful for understanding how to create your param grid.
#    # grid.get_params().keys()
#    
#    grid.fit(X,y)

#%% 6. Define Model(naive Bayes)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
vect = TfidfVectorizer(analyzer='word',
                       #stop_words='english',
                       max_features = 850, ngram_range=(1, 1),
                       binary=False, lowercase=True, norm=None, smooth_idf=True, strip_accents=None,
                       sublinear_tf=True, use_idf=False)

mcl_transformed = vect.fit_transform(X)
#vectNames = vect.get_feature_names()

nb_SP_Model = MultinomialNB()
nb_SP_Model.fit(mcl_transformed, y)
print ("Model accuracy within dataset: ", nb_SP_Model.score(mcl_transformed, y))

print ("Model accuracy with cross validation:", cross_val_score(MultinomialNB(), mcl_transformed.toarray(), 
                                                                y, cv=5, scoring="accuracy").mean())

#%% 6. Define Model(LDA)
# Vectorization
from sklearn.feature_extraction.text import CountVectorizer
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                ngram_range=(1, 2),
                                stop_words='english')
                                #max_features=850)
tf = tf_vectorizer.fit_transform(X)

# Fit LDA Model
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=len(top_speakers),
                                #doc_topic_prior = 10,
                                learning_offset=50.,
                                random_state=0)
docres = lda.fit_transform(tf)

#%% 6.0.1 Count Prediction Distribution
allScriptsTF = tf_vectorizer.transform(X)
allPredIDArr = np.argmax(lda.transform(allScriptsTF),axis=1)
allCounter = collections.Counter(allPredIDArr)

yCounter = collections.Counter(y)

#%% 6.1 Label Alignment for LDA
mapDictFromTrueToCluID = dict()
mapDictFromCluToTrueID = dict()
import collections
for speakerIdx in range(len(top_speakers)):
    # Speaker true ID and corresponding scripts
    speakerTrueID = top_speakers.index[speakerIdx]
    speakerScripts = main_char_lines[main_char_lines['character_id']==speakerTrueID]['normalized_text']
    # Vectorization and predict
    speakerScriptsTF = tf_vectorizer.transform(speakerScripts)
    speakerPredIDArr = np.argmax(lda.transform(speakerScriptsTF),axis=1)
    # Find majority prediction
    counter = collections.Counter(speakerPredIDArr)
    speakerPredID = str(counter.most_common(1)[0][0])
    # Update mapping dict
    mapDictFromTrueToCluID[speakerTrueID] = speakerPredID
    mapDictFromCluToTrueID[speakerPredID] = speakerTrueID

#%% 6.2 Accuracy(LDA)
predictSpeakerScriptsTF = tf_vectorizer.transform(X)
predictSpeakers = np.argmax(lda.transform(predictSpeakerScriptsTF),axis=1)
correctCount = 0
ylist = list(y)
for scriptIdx in range(len(X)):
    speakerTrueID = ylist[scriptIdx]
    speakerPredIDAligned = mapDictFromTrueToCluID[speakerTrueID]
    if speakerPredIDAligned == str(predictSpeakers[scriptIdx]):
        correctCount += 1
acc = correctCount/len(X)
print('ACC of LDA: ', str(acc))

#%% 7. Test Model
# Predict on new text
#new_text = ["Well, I guess we'll have to roshambo for it. I'll kick you in the nuts as hard as I can, then you kick me square in the nuts as hard as you can..."]
#new_text_transform = vect.transform(new_text)
#
#print (nb_SP_Model.predict(new_text_transform)," most likely said it.")