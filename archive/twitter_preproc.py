# -*- coding: utf-8 -*-
"""
Twitter Topic-Account Analysis

Using data from "24 thousand tweets later-2017 tweets from incubators and accelerators"
> https://www.kaggle.com/derrickmwiti/24-thousand-tweets-later
"""

#%% 1. Import Data
import numpy as np
import pandas as pd
twitter_data_raw = pd.read_csv('../data/tweets_incubators_accelerators.csv',
                                    error_bad_lines=False,
                                    warn_bad_lines=False,
                                    low_memory=False)

#%% Simple Statistics
from collections import Counter
cnt = Counter(twitter_data_raw['username'])
topFreqAccountNum = 10
topFreqAccouts = cnt.most_common(topFreqAccountNum)
topFreqAccoutsNames = [w[0] for w in topFreqAccouts]
topFreqData_raw = twitter_data_raw[twitter_data_raw['username'].isin(topFreqAccoutsNames)]
"""
('Seedstars', 3248),
('TonyElumeluFDN', 2622),
('MESTAfrica', 2184),
('ActiveSpaces', 1949),
('Cc_HUB', 1733),
('BongoHive', 1698),
('thenailab', 1664),
('Afrilabs', 1453),
('Sbootcamp', 1423),
('iHub', 1274)

Seedstars: innovation in emerging markets
TonyElumeluFDN: African-based African-funded philanthropic organisation. Founded in 2010, our mission is to support entrepreneurship in Africa.
MESTAfrica: Training, investing and incubating African tech entrepreneurs. @FastCompany's Top 10 Most Innovative Companies in Africa.
ActiveSpaces: Cannot find
Cc_HUB: Open living lab for entrepreneurs, investors, tech companies and hackers in and around Lagos. A cushy nest and co-working space for social tech ventures.
(Tech)
BongoHive: Lusaka's Technology & Innovation Hub.
thenailab: Business incubator focused on providing the right ingredients to turn business ideas into viable startups.
(startup)
AfriLabs: AfriLabs is a network organization for the growing number of Africa based technology and innovation hubs.
Sbootcamp: Accelerating Innovators | 20+ Programs | 1 Powered-By Program | 6 Continents | Never miss a beat, sign-up to our newsletter 👉 http://eepurl.com/XbfWH 
iHub: Nairobi's Innovation Hub.
"""    

#%% 2. Data Prepeocessing
# 2.1 Remove URLs (e.g. https://t.co/ZUOJ92uRNo)
# 2.2 Lower and remove punctuations
# 2.3 (Optional) Remove @user
# 2.4 (Not included)Record String Length
# 2.5 Record Spliting Parts: mainBody(within ""), tags(begin with #), atUsers(begin with @)

smalldf = topFreqData_raw.iloc[:5]  # for test
df = pd.DataFrame(columns = ['username','strMain', 'strTags','strAtUsers','retweets'])
import re

for twIdx, row in topFreqData_raw.iterrows():
#for twIdx, row in smalldf.iterrows():
    # read raw string
    strRaw = row['tweet ']
    
    # remove url
    string = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', strRaw)
    
    # split into tags, atUsers and mainBody 
    strList = re.sub(r'"',' ',string).split()
    # re.sub(r'[^\w\s]','',word): remove all punctuations from string word
    strTags = [re.sub(r'[^\w\s]','',word) for word in strList if word.startswith('#')]
    strAtUsers = [re.sub(r'[^\w\s]','',word) for word in strList if word.startswith('@')]    
    strExtra = []; strExtra.extend(strTags); strExtra.extend(strAtUsers)
    strMain = [re.sub(r'[^\w\s]','',word) for word in strList if re.sub(r'[^\w\s]','',word) not in strExtra]
    
    # don't store empty ones
    if len(strMain) > 0:
        df = df.append({'username':row['username'],'strMain':strMain, 'strTags':strTags,'strAtUsers':strAtUsers,'retweets':row['retweets']},ignore_index=True)

#%% 3. Save Preproceed Data
dff = df.copy()
dff = dff.rename(index=str, columns={'expStrMain':'strMainList', 'expStrTags':'strTagsList', 'expStrAtUsers':'strAtUsersList'})
dff.to_csv('../data/tweets_incubators_accelerators_prepro.csv', index=False)