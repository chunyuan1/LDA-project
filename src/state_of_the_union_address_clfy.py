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

#%% 1.1 Choose Presidents
# prsdList = ['Bush1','Cliton','Bush2','Obama','Trump']
# prsdList = ['Bush1','Trump']
# data = data[[(prsd in prsdList) for prsd in data['president']]]

#%% 2 Set Parameters
dimList = [5, 10, 20]#, 50]
alphaList = [1e-4, 1e-2, 1, 10]#, 50]
testSizeList = [0.1, 0.3, 0.5, 0.8]#, 0.9]
usingTfiDFList = [0]#, 1]
rerunNum = 3
cvFoldNum = 3

import pandas as pd
from state_of_the_union_address_clfy_func import state_of_the_union_address_clfy_func
records = pd.DataFrame(columns = ['dim', 'alpha', 'test_size', 'method', 'usingtfidf', 'CV_score', 'test_score'])

#%% 3. Run 
ifrun = False
if ifrun:
    for dim in dimList:
        for alpha in alphaList:
            for test_size in testSizeList:
                for usingTfiDF in usingTfiDFList:
                    record = state_of_the_union_address_clfy_func(data = data, test_size=test_size, dim = dim, alpha=alpha, usingTFiDF=usingTfiDF, cvFoldNum=cvFoldNum, rerunNum=rerunNum, verbose = 1)
                    records = pd.concat([records, record])
else:                    
    records = pd.read_csv('../data/address_records_part.csv')
records = records[(records['dim']!=50)&(records['test_size']!=0.9)&(records['usingtfidf']==0)]
# records.to_csv('../data/address_record_small.csv')
                
                
"""
Guess: LDA works better than others when:
1. Smaller dataset
2. Smaller alpha
3. Lower dimension
"""

#%% Plot Function
def plotScore(recordDim, title='hahaha', xlabel='Model', ylabel='Accuracy'):    
    import numpy as np
    import matplotlib.pyplot as plt
    n_groups = len(recordDim)
    names = [item['method'] for idx, item in recordDim.iterrows()]
    CVs = [item['CV_score'] for idx, item in recordDim.iterrows()]
    Tests = [item['test_score'] for idx, item in recordDim.iterrows()]    
        
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
         
    opacity = 0.4    
    rects1 = plt.bar(index, CVs, bar_width, alpha=opacity, color='b',label='10-fold Cross Validation')
    rects2 = plt.bar(index + bar_width, Tests, bar_width, alpha=opacity, color='r', label='Test Accuracy')    
    # rects3 = plt.bar(index + 2*bar_width, times, bar_width, alpha=opacity, color='g', label='Time(s)')
    
    def autolabel(rects):
    #Attach a text label above each bar displaying its height    
        for rect in rects:
            #height = round(rect.get_height()*10000)/100
            height = rect.get_height()
            # print(height)
            ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                    #'%d' % int(height),
                    str(int(height*10000)/100),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    # autolabel(rects3)
        
    plt.xlabel(xlabel)    
    plt.ylabel(ylabel)    
    plt.title(title)
    plt.xticks(index + bar_width -0.17, names)    
    plt.legend();
    plt.ylim([0,1])
      
    plt.tight_layout();       
    plt.show();       

#%% 4.1 Analysis: dim: does LDA works better (than others) with lower dim?
# Fix: alpha=1e-4, test_size=0.3, 0.8
# Change: dimension
alpha = alphaList[1]
test_size=0.8
#import numpy as np
#import matplotlib.pyplot as plt
for dim in dimList:
    recordDim = records[(records['alpha']==alpha)&(records['test_size']==test_size)&(records['dim']==dim)][['dim', 'method', 'CV_score','test_score']]
    plotScore(recordDim, title = 'Average 10-rerun Accuracies, dimension={}, alpha={}, test_size={}'.format(dim, alpha, test_size))
    

#%% 4.2 Analysis: test_size: does LDA performs better with small data?
# Think peusdo-count
alpha= alphaList[1]
dim = 5
for test_size in testSizeList:
    recordTS = records[(records['alpha']==alpha)&(records['test_size']==test_size)&(records['dim']==dim)][['dim', 'method', 'CV_score','test_score']]
    plotScore(recordTS, title = 'Average 10-rerun Accuracies, dimension={}, alpha={}, test_size={}'.format(dim, alpha, test_size))

#%% 4.3 Analysis: alpha: does prior matters?                
# Show example: politics, food, ... instead european pol, US pol, ...
dim = 5
test_size=0.3
for alpha in alphaList:
    recordAlpha = records[(records['alpha']==alpha)&(records['test_size']==test_size)&(records['dim']==dim)][['dim', 'method', 'CV_score','test_score']]
    plotScore(recordAlpha, title = 'Average 10-rerun Accuracies, dimension={}, alpha={}, test_size={}'.format(dim, alpha, test_size))
    
