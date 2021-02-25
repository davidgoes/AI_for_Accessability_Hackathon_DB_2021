#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:07:27 2021

@author: DavidGoes
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediciting Credit Risk with Open Text Data
@author: DavidGoes(6767512)
"""

# =============================================================================
# Sentiment Analysis - dictionary-based - VADER 
# =============================================================================

import pandas as pd
import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
pd.options.mode.chained_assignment = None



df_tweets_all = pd.read_csv('')

print('Tweets loaded')

#Drop empty Tweets
df_tweet_all_dropped = df_tweets_all.dropna(subset=['Tweet'])
df_tweets_all_shape = df_tweets_all.shape #drop missings
#df_tweet_all_dropped.shape

#pre-processing
#Stop words
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()


#Preprocess Tweets
def normalizer(tweet):
    
    #removing username, cahsticker and hashtag
    only_letters = re.sub('@[^\s]+', '', tweet) #removing username
    only_letters = re.sub('\$[^\s]+', '', only_letters) #removing cashticker
    only_letters = re.sub(r'#([^\s]+)', r'\1', only_letters) #removing #, leaving the word
    
    #removing links
    only_letters = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', only_letters)
    only_letters = re.sub(r'http\S+', '', only_letters)
    
    #removing remaining special signs and number; dont remove things before url 
    only_letters= re.sub('[0-9]', '', only_letters) 
    only_letters= re.sub('[$_@.&+#?:;.,\'\"\-%!*\(\)]', '', only_letters) 

    #lowercase, remove stopwords
    lower_case_letters = only_letters.lower()
    tokens = nltk.word_tokenize(lower_case_letters)
    filtered_result = list(filter(lambda l: l not in stop_words, tokens))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    
    #join tokens for VADER
    tweet = ', '.join(lemmas)
    
    return tweet #only_letters#lower_case_letters #only_letters filtered_result

#Pre-processing Tweets
df_tweets_all.loc[:, 'Tweets_clean'] = df_tweets_all.Tweet.apply(lambda x: normalizer(x))

#Clean data frame
df_tweets_all = df_tweets_all [df_tweets_all .Tweets_clean.isna() != True]\
    .reset_index(drop=True)
df_tweets_all['ID'] = df_tweets_all.index

print('Tweets preprocessed, start sentiment analysis')


#Save final data frame as csv
path  = '' # set path indiviually
#df_tweets_all.to_csv(path)

print('New data set has been stored in: ', path)


