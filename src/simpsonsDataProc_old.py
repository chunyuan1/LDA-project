"""
Data Preprocessing for Simpsons,
    extarcting all scripts

See more:
# https://www.kaggle.com/rivanor/wordclouds-from-springfield-people
"""
def simpPreProcLoc(filePath = '../data/simpsons_script_lines.csv', remove_stop_words = False, getId=False, saveCSV = False, textType='normalized_text'):
    # 1. Read File
    import numpy as np
    import pandas as pd    
    data_script_lines = pd.read_csv(filePath,
                                    error_bad_lines=False,
                                    warn_bad_lines=False,
                                    low_memory=False)
    
    # 2. Extract Cols and Rows
    # Exrtact lines which is conversations (not cmomments)
    data_true_script_lines = data_script_lines[data_script_lines['speaking_line']=='true']
    data_script_rows = data_true_script_lines[textType].tolist()
    
    data_script_rows_character = data_true_script_lines['location_id'].fillna(0).astype(int)#.tolist()
    data_script_rows_character = pd.to_numeric(data_script_rows_character, downcast='integer').tolist()
    
    # 3. PreProcessing
    import nltk
    from nltk.stem.wordnet import WordNetLemmatizer 
    lem = WordNetLemmatizer()
    
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    stop_words = set(stopwords.words('english'))
    for idx, row in enumerate(data_script_rows):        
        row = str(row)
        # Normalize
        """
        row_split = nltk.tokenize.word_tokenize(row)
        for idx in range(len(row_split)):
            row_split[idx] = lem.lemmatize(row_split[idx])
        row = ' '.join(row_split)
        """
        # Remove stop words
        if remove_stop_words:
            word_tokens = word_tokenize(row)
            row = [w for w in word_tokens if not w in stop_words]
            row = ' '.join(row)
        data_script_rows[idx] = row

    # 4. Save to CSV file
    if saveCSV:
        if getId:
            df = pd.DataFrame({'character_id':data_script_rows_character, 'script':data_script_rows})
            df.to_csv('../data/simpsons_loc_script_lines_extracted.csv', index=False)
        else:
            ss = pd.Series(data_script_rows)
            ss.to_csv('../data/simpsons_loc_script_lines_extracted.csv', index=False)
        
    # 5. Return
    if getId:
        return data_script_rows, data_script_rows_character
    else:
        return data_script_rows

def simpPreProc(filePath = '../data/simpsons_script_lines.csv', remove_stop_words = False, getId=False, saveCSV = False, textType='normalized_text'):
    # 1. Read File
    import numpy as np
    import pandas as pd    
    data_script_lines = pd.read_csv(filePath,
                                    error_bad_lines=False,
                                    warn_bad_lines=False,
                                    low_memory=False)
    
    # 2. Extract Cols and Rows
    # Exrtact lines which is conversations (not cmomments)
    data_true_script_lines = data_script_lines[data_script_lines['speaking_line']=='true']
    data_script_rows = data_true_script_lines[textType].tolist()
    
    data_script_rows_character = data_true_script_lines['character_id'].fillna(-1).astype(int)#.tolist()
    data_script_rows_character = pd.to_numeric(data_script_rows_character, downcast='integer').tolist()
    
    # 3. PreProcessing    
    import nltk
    from nltk.stem.wordnet import WordNetLemmatizer 
    lem = WordNetLemmatizer()
    
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    stop_words = set(stopwords.words('english'))
    
    #stop_words.update()
    new_sd = ['oh','well','im','the','you','i','yes','one','dont','see','--','go','look','make','like','youre','thats','ive','take','good','say','get','going']
    for wordIdx in range(len(new_sd)):
        stop_words.add(new_sd[wordIdx])
    for idx, row in enumerate(data_script_rows):
        row = str(row)
        # Normalize
        
        row_split = nltk.tokenize.word_tokenize(row)
        for idx in range(len(row_split)):
            row_split[idx] = lem.lemmatize(row_split[idx],'v')
        row = ' '.join(row_split)
        
        # Remove stop words
        if remove_stop_words:
            word_tokens = word_tokenize(row)
            row = [w for w in word_tokens if not w in stop_words]
            row = ' '.join(row)
        data_script_rows[idx] = row
    
    
    # 4. Save to CSV file
    if saveCSV:
        if getId:
            df = pd.DataFrame({'character_id':data_script_rows_character, 'script':data_script_rows})
            df.to_csv('../data/simpsons_script_lines_extracted.csv', index=False)
        else:
            ss = pd.Series(data_script_rows)
            ss.to_csv('../data/simpsons_script_lines_extracted.csv', index=False)
        
    # 5. Return
    if getId:
        return data_script_rows, data_script_rows_character
    else:
        return data_script_rows

# Set IDs of all non-common/rare characters to 0
def ignoreRareChar(scripts_id, commonCount = 30):
    import collections
    counter = collections.Counter(scripts_id)
    mostCommonCharId = counter.most_common(commonCount)
    mostCommonCharId = [i[0] for i in mostCommonCharId]
    for idx, charaIdx in enumerate(scripts_id):
        if charaIdx in mostCommonCharId:
            pass
        else:
            scripts_id[idx] = 0
    return scripts_id, mostCommonCharId
    
    
def concatenateScriptListByCharacter(scripts, scripts_id, saveCSV = False):#, ignoreRare = True):    
    # Init dict
    scriptsByCharDict = dict()
    for idx, charaID in enumerate(scripts_id):
        scriptsByCharDict[str(charaID)]=''
    
    # Concatenate stripts of same character
    for idx, charaID in enumerate(scripts_id):
        #print(scripts[idx] + ' ')
        scriptsByCharDict[str(charaID)] += (str(scripts[idx]) + ' ')
    
    scriptsByCharList = list(scriptsByCharDict.values())
    
    # Save file
    if saveCSV:
        import pandas as pd
        ss = pd.Series(scriptsByCharList)
        ss.to_csv('../data/simpsons_script_lines_grouped.csv', index=False)
        
    return scriptsByCharList