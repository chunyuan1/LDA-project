"""
Data Preprocessing for Simpsons,
    extarcting all scripts

See more:
# https://www.kaggle.com/rivanor/wordclouds-from-springfield-people
"""
filePath = '../data/simpsons_script_lines.csv'
remove_stop_words = False
getId=False
saveCSV = False
textType='normalized_text'

#%% 1. Read File
import numpy as np
import pandas as pd    
data_script_lines = pd.read_csv(filePath,
                                error_bad_lines=False,
                                warn_bad_lines=False,
                                low_memory=False)

data_true_script_lines = data_script_lines[data_script_lines['speaking_line']=='true']
data_true_script_lines = data_true_script_lines[['character_id', 'raw_character_text', 'normalized_text', 'word_count']]

data_true_script_lines = data_true_script_lines[data_true_script_lines['word_count']>=10]

#%% 2. Keep only Active Caharcters
commonCount = 2

import collections
counter = collections.Counter(data_true_script_lines['character_id'])

mostCommonCharId = counter.most_common(commonCount)
mostCommonCharId = [i[0] for i in mostCommonCharId]

data_true_script_lines_active = data_true_script_lines[[charID in mostCommonCharId for charID in data_true_script_lines['character_id']]]

#%% 3. Remove stop words
#from nltk.corpus import stopwords
#stop_words = set(stopwords.words('english'))
#for idx, row in data_true_script_lines_active.iterrows():
#    text = row['normalized_text']
#    # Remove stop words
#    text = ' '.join([word for word in text.split() if word not in stop_words])
#    data_true_script_lines_active.iloc[[idx]]['normalized_text'] = text


#%% 4. Save to CSV file
data_true_script_lines_active.to_csv('../data/simpsons_proc_{}.csv'.format(commonCount), index=False)