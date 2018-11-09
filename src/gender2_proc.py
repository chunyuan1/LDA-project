"""
Gender classification using blog data

"""
#%% 1. Read data
import pandas as pd
from sklearn.model_selection import train_test_split
test_size = 0.5
blog = pd.read_csv('../data/blog-gender-dataset-clean.csv',error_bad_lines=False, header=None)
blog.columns = ['gender','blog']
# blog_train, blog_test = train_test_split(blog, random_state=1, test_size = test_size)

#%% 2. Processing: lower, normalize, remove stop words
# Set Stop Words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
swFile = open("stop_words.txt")
swFromFile = swFile.read().split()
swFile.close();     stop_words += swFromFile
stop_words.append('')
stop_words.append('-')
stop_words.append('--')

# Filter out all non-ascii characters
isascii = lambda s: len(s) == len(s.encode())

import re
import nltk
lem = nltk.stem.wordnet.WordNetLemmatizer()
data = pd.DataFrame(columns = blog.columns)
for scrIdx, row in blog.iterrows():
    # read raw string and lowering
    strWordlist = row['blog'].lower().split()
    
    # remove all punctuations from string word
    strWordlist = [re.sub(r'[^\w\s]','',word) for word in strWordlist]
    
    # Remove stopWords and normalize as noun
    strWordlist = [lem.lemmatize(word,'n') for word in strWordlist if word not in stop_words]
    
    # Remain onyl nouns
    # strMain = [word for (word, pos) in nltk.pos_tag(strMain) if is_noun(pos)] 
    
    # Save preprocessed data
    data = data.append({'gender':row['gender'], 'blog':' '.join(strWordlist)},ignore_index=True)
data.to_csv('../data/blog-gender-dataset-proc.csv',index=False)