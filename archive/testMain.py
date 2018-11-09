#%% 1. Read data and preprocessing
import numpy as np
from simpsonsDataProc import simpPreProc, concatenateScriptListByCharacter, ignoreRareChar
scripts, scripts_id = simpPreProc(getId = True, remove_stop_words = True)
scripts_id = np.array(scripts_id)

#%% 2. Turn all uncommon characters index to 0
commonCount = 30
scripts_id, mostCommonCharId = ignoreRareChar(scripts_id, commonCount = commonCount)
scripts_id_common = scripts_id[scripts_id!=0]

# show the most common ones' names
import pandas as pd
chars = pd.read_csv('../data/simpsons_characters.csv')
for scrIdx in range(commonCount):
    charID = mostCommonCharId[scrIdx]
    charName = (chars.loc[chars['id'] == charID])['name'].iloc[0]
    print('The '+str(scrIdx)+' talk holic person is '+charName)

for idx in range(len(scripts)-1,-1,-1):
    if scripts_id[idx]==0:
        del scripts[idx]

#%% 3. Concatenate Script Dict By Character
# scriptsByCharList[1] = a long string contarining all scripts of character scripts_id[1], separated by SPACE
scriptsByCharList = concatenateScriptListByCharacter(scripts, scripts_id_common)

#%% 4. sLDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
cntVector = CountVectorizer()
cntTf = cntVector.fit_transform(scriptsByCharList)

lda = LatentDirichletAllocation(n_components=commonCount,
                                learning_offset=50.,
                                random_state=0)
docres = lda.fit_transform(cntTf)

#%% 5. Show 
for scrIdx in range(commonCount):
    charID = mostCommonCharId[np.argmax(docres[0])]
    charName = (chars.loc[chars['id'] == charID])['name'].iloc[0]
    print('The '+str(scrIdx)+' script is spoken by '+charName)