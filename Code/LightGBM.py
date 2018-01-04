# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Any results you write to the current directory are saved as output.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def getdata():
    train = pd.read_csv('../input/train.tsv', sep = "\t")
    test = pd.read_csv('../input/test.tsv', sep = "\t")
    return train, test

def clean(data):
    #cleaning data & converting to strings for countvectorizer
    data['general_cat'] = data['general_cat'].fillna('missing').astype(str)
    data['subcat_1'] = data['subcat_1'].fillna('missing').astype(str)
    data['subcat_2'] = data['subcat_2'].fillna('missing').astype(str)
    data['brand_name'] = data['brand_name'].fillna('missing').astype(str)
    data['shipping'] = data['shipping'].astype(str)
    data['item_condition_id'] = data['item_condition_id'].astype(str)
    data['item_description'] = data['item_description'].fillna('None')
    
    return data

def split_Cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")

def tokenise(train, test):
    default_preprocessor = CountVectorizer().build_preprocessor()
    
    def build_preprocessor(field):
        field_idx = list(train.columns).index(field)
        return lambda x: default_preprocessor(x[field_idx])
    
    vectorizer = FeatureUnion([
    ('name', CountVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        preprocessor=build_preprocessor('name'))),
    ('general_cat', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('general_cat'))),
    ('subcat_1', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('subcat_1'))),            
    ('subcat_2', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('subcat_2'))),            
    ('brand_name', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('brand_name'))),
    ('shipping', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('shipping'))),
    ('item_condition_id', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('item_condition_id'))),
    ('item_description', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=50000,
        stop_words='english',
        preprocessor=build_preprocessor('item_description'))),
    ])
    
    print('features creation completed')
    
    X_train = vectorizer.fit_transform(train.values)
    X_test = vectorizer.fit_transform(test.values)

    return X_train, X_test

def main():
    train, test = getdata()
    print('loaded data')
    
    train['general_cat'], train['subcat_1'], train['subcat_2'] = \
        zip(*train['category_name'].apply(lambda x: split_Cat(x)))
    
    test['general_cat'], test['subcat_1'], test['subcat_2'] = \
        zip(*test['category_name'].apply(lambda x: split_Cat(x)))
    
    train.drop('category_name', axis=1, inplace=True)
    test.drop('category_name', axis=1, inplace=True)
    
    train = clean(train)
    train = train.drop(train[train.price == 0].index)
    y = np.log1p(train['price'])
    train.drop(['price'], axis=1,inplace=True)
    test = clean(test)
    print('Cleaned data')
    
    X, X_test = tokenise(train, test)
    
    print('Tokenised Completed')
    
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.15, random_state = 42) 
    
    #LightGBM Setup
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(valid_X, label=valid_y)
    watchlist = [d_train, d_valid]
    
    params = {
        'learning_rate': 0.65,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 60,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.5,
        'nthread': 4
    }
    
    model = lgb.train(params, train_set=d_train, num_boost_round=8000, valid_sets=watchlist, \
    early_stopping_rounds=1000, verbose_eval=1000) 
    print('Modelling training complete')
    
    predicted_price = model.predict(X_test)

    submission= pd.DataFrame(test[['test_id']])
    submission['price'] = predicted_price
    submission.to_csv("submission.csv", index=False)
    print('Completed')

if __name__ == '__main__':
    main()