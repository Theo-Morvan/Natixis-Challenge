#!/usr/bin/env python
# coding: utf-8

# In[1]:

from sklearn.pipeline import Pipeline
import nltk
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from nltk.stem.snowball import SnowballStemmer
import regex
import re
import shutil
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.manifold import TSNE
import hexuity as hx


import string
import fasttext
import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from deep_translator import GoogleTranslator
from tqdm import tqdm

# load a pretrained model from facebooks ai team - wooow
# fmodel = fasttext.load_model('./data/lid.176.ftz')

# removes all non-letters
def stripper(dirty):
    try:
        return ''.join([x for x in dirty if x in string.ascii_letters + '\'- '])
    except:
        return ''

# reduces df to series for faster stemming
def one_series(X):
    X = X.copy()
    x = pd.Series()
    for col in X.columns:
        x = x.append(X.loc[:,col])
    x = x.drop_duplicates()

    x.replace('', np.nan, inplace=True)
    x.dropna(inplace=True)

    return x

# creates a stemming dictornary from series
def series_stemmer(X, translate=True):

    # creates index for later matching
    index = X.apply(lambda x: x[:30].strip())

    # creates df with speeches and their languages
    stemmed = pd.DataFrame(X.copy())
    stemmed.columns = ['speeches']
    stemmed['language'] = stemmed['speeches'].apply(
        lambda x: fmodel.predict(x.replace('\n',' '))[0][0][-2:])

    # manages non-english
    for i in tqdm(range(len(stemmed))):
        # checks whether eng or not
        if stemmed.iloc[i, 1] != 'en':
            # checks whether to translate
            if translate:
                # ungly splits to get aroubd 5000 char limit...
                splits = int(len(stemmed.iloc[i, 0])/5000) + 1
                speech = ''
                for j in range(splits):
                    speech = speech + ' ' + GoogleTranslator(
                        source='auto', target='en').translate(
                            stemmed.iloc[i, 0][j*4999:(j+1)*4999])
                stemmed.iloc[i, 0] = speech
            # drops non-english if translate = False
            else:
                stemmed.iloc[i, 0] = ''

    stemmed = stemmed['speeches']
    stemmer = SnowballStemmer('english')

    stemmed = stemmed.apply(lambda element : element.split(" ")
        ).apply(lambda x: [stemmer.stem(y) for y in x]
        ).apply(lambda x: ' '.join([item for item in x]))

    output = pd.Series(stemmed.values, index=index).to_dict()
    output[''] = ''
        
    return output

def stemmer(X, translate=True):
    X = X.copy()

    # clean out all non-letters
    X = X.applymap(stripper)

    # create stem dictornary
    one_row = one_series(X)
    stem_dict = series_stemmer(one_row, translate=translate)

    # apply stemming to df
    X = X.applymap(lambda x: x[:30].strip())
    X = X.applymap(lambda x: stem_dict[x])

    return X

os.chdir('C:/Users/32mor/OneDrive/Documents/Polytechnique/NLP & Natixis/starting_kit/data/data/train')
data_train = pd.read_json(os.listdir()[1])
speeches = pd.DataFrame(data_train['speech'].values.tolist())
speech_cols = pd.DataFrame(index=speeches.index)


def load_speech_dataset(X, bank, index='vix'):
    # speeches,_,_ = hx.data_expander(price_data=True, all_data=True,index=index)
    speeches = X.copy()
    speeches_bank = speeches.filter(like=bank)
    return speeches_bank

def _transform_data_for_tfidf(X):
    '''Function that transforms a pandas Serie to a list. Used prior to the TfIdfVectorizer to increase its stability
    args :
        -X : output of _transform_nlp_data
    returns :
        - corpus : list of tokenized ans stemmed speeches.'''

    X = X.copy()
    corpus = X.melt().iloc[:,1].to_list()
    # corpus = list(X.iloc[:,0].tolist())
    return corpus

def _tf_idf_results_pandas(X):
    X = X.copy()
    X = pd.DataFrame(X.toarray())
    shape = X.shape
    columns_data = [f'colonne_{i}' for i in range(1, shape[1]+1)]
    X.columns = columns_data
    return X

def _sparse_matrix_toarray(X):
    '''Function needed to transform a sparse matrix to a dense numpy array. Required for dimension reduction and scaling.'''
    X = X.copy()
    return X.toarray()

def Full_pipeline_nlp(bank, max_df, components,translate):
    '''Sklearn pipeline which fully transforms the speeches to a numerical embedding thanks to tf-idf and dimension reduction.
    args :
        - bank : the central bank whose speeches will be transformed
        - max_df : hyperparameter in TfIdf which drops tokens or n-grams present in the corpus of documents more than the given percent
        - components : new dimension after dimension reduction in pca or t-sne
    returns :
        - pipe : sklearn pipeline ready to be used to transform the data.'''
    max_variables = 10_000
    colonnes = [f'colonne_{i}' for i in range(1, max_variables+1)]

    
    speeches_loader = FunctionTransformer(load_speech_dataset, validate=False, kw_args={'bank':bank})
    stemmer_preprocessor = FunctionTransformer(stemmer, validate=False, kw_args={'translate':translate})

    # language_preprocessor = FunctionTransformer(language_transformer,kw_args = {'bank': bank, 'nlp': nlp},validate=False)
    # speeches_formator = FunctionTransformer(form_speeches_dataset, validate=False)
    # nlp_preprocessor = FunctionTransformer(_transform_nlp_data, kw_args = {'bank': bank} , validate=False)
    tf_idf_preprocessor = FunctionTransformer(_transform_data_for_tfidf, validate=False)
    sparse_to_dense = FunctionTransformer(_sparse_matrix_toarray, validate=False)
    
    # preprocessor = ColumnTransformer(
    #     [('standard-scaler', StandardScaler(), colonnes)])
    max_df = 0.8
    components = 300
    bank = 'bce'
    # il nous faut un pipeline qui prend le dataset en ent
    pipe = Pipeline([
        # ('non_english_speeches', language_preprocessor())---de là
        # ('forming_speeches', speeches_formator),
        # ('nlp_processing', nlp_preprocessor), ---> erik's pipeline
        ('loading_speechees',speeches_loader),
        ('stemming_speeches',stemmer_preprocessor),
        ('tf_idf_preprocessor', tf_idf_preprocessor),
        ('tf_idf_vectorization', TfidfVectorizer(stop_words = 'english', ngram_range=(3,3), max_features = 10_000, max_df=max_df)),
        ('sparse_to_dense', sparse_to_dense),
        ('scaler', StandardScaler()),
        ('dimension_reduction', PCA(n_components=components))
        #('lstm_reshape', )
        ])
   
 

    return pipe


def load_numerical_data(X):
    # speeches,_,_ = hx.data_expander(all_data=True,index=index) --> X
    numerical_data = X.filter(like='price')
    return numerical_data
    
def EMA(X): # implement baseline strategy of ema
    X = X.copy()
    a = 2 / (X.shape[1] +1)
    X['ema1'] = X['price1']
    for i in range(2,21):
        X['ema{}'.format(i)] = (
            a * X['price{}'.format(i)] 
            + (1 - a) * X['ema{}'.format(i-1)])
    
    return X


def _numerical_transformation(X, ema, lstm):
    X = X.copy()
    depth = 1
    day_max = X.shape[1]
    if ema:
        X = EMA(X)
        depth += 1
    
    if lstm:
        # day_max = X.shape[1]
        speeches_max = X.shape[0]
        X = np.expand_dims(np.array(X), 
                            axis=1).reshape((speeches_max,day_max,depth))
    
    return X
# def _numerical_transformation(X):
#     X = X.copy()
#     day_max = X.shape[1]
#     speeches_max = X.shape[0]
    
#     X_lstm_num = np.expand_dims(np.array(X), 
#                         axis=1).reshape((speeches_max,day_max,1))
#     # dataset_num = pd.concat([X, X_ema],axis=1)
#     # if format == 'lstm' : X_lstm_num[:,:,1] = X_ema
#     # X_ema = function_ema(X)
#     # X_lstm_num[:,:,1] = X_ema
#     return X_lstm_num
    
def numerical_pipeline(ema=True,lstm=True):
    num_loader = FunctionTransformer(load_numerical_data,validate=False)
    num_transformer = FunctionTransformer(_numerical_transformation, validate=False, kw_args={'ema':True,
                            'lstm':True})

    pipe = Pipeline([('loading_numerical_data',num_loader),
        ('num_processing', num_transformer)])
    
    return pipe

class full_pipeline:

    def __init__(self, fed_pipe, ecb_pipe, num_pipe):
        self.fed_pipe = fed_pipe
        self.ecb_pipe = ecb_pipe
        self.num_pipe = num_pipe

    def fit(self, X):
        self.fed_pipe.fit(X)
        self.ecb_pipe.fit(X)
        self.num_pipe.fit(X)
    
    def fit_transform(self, X):
        X_fed = self.fed_pipe.fit_transform(X)
        X_ecb = self.ecb_pipe.fit_transform(X)
        X_num = self.num_pipe.fit_transform(X)

        return X_fed, X_ecb, X_num

    def transform(self, X):
        X_fed = self.fed_pipe.transform(X)
        X_ecb = self.ecb_pipe.transform(X)
        X_num = self.num_pipe.fit_transform(X)

        return X_fed, X_ecb, X_num

#X_fed --> numpy array avec 1 ligne par discours par jour et 300 colonnes.
#Dataframe qui a 1 ligne par input et 300 * 20 colonnes pour la bce --> PCA et tu concatène avec les données numériques, bce et ema


def multi_input_output_lstm_full(shape_nlp, shape_num):
    input_nlp_fed = Input(shape=(shape_nlp[0], shape_nlp[1]))
    input_nlp_bce = Input(shape=(shape_nlp[0], shape_nlp[1]))
    input_num = Input(shape=(shape_num, 1))


    w = LSTM(100, return_sequences=True)(input_nlp_fed)
    w = Dropout(0.3)(w)
    w = BatchNormalization()(w)
    w = LSTM(40, return_sequences=False)(w)
    #w = Dropout(0.3)(w)
    w = Model(inputs=input_nlp_fed, outputs=w)


    x = LSTM(100, return_sequences=True)(input_nlp_bce)
    x = Dropout(0.3)(x)
    #x = BatchNormalization()(x)
    x = LSTM(40, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    x = Model(inputs=input_nlp_bce, outputs=x)

    # the second branch opreates on the second input
    y = LSTM(100, return_sequences=True)(input_num)
    y = Dropout(0.3)(y)
    #y = BatchNormalization()(y)
    y = LSTM(40, return_sequences=False)(y)
    y = Dropout(0.3)(y)
    y = Model(inputs=input_num, outputs=y)
    # combine the output of the two branches
    combined = concatenate([w.output, x.output, y.output])

    z = Dense(256, activation='relu')(combined)
    z = Dense(64, activation='relu')(z)
    out_reg = Dense(1, activation='linear')(z)
    # dense_class = Dense(40, activation = 'relu')(dropout_2)
    out_class = Dense(1, activation = 'sigmoid')(z)
    model = Model(inputs=[w.input, x.input, y.input], outputs=[out_reg, out_class])

    return model

def homemade_train_test_split(len_dataset, train_size):
    idx = np.arange(len_dataset)
    np.random.shuffle(idx)
    stop_train_idx = int(0.8 * len_dataset)
    train_idx = idx[:stop_train_idx]
    test_idx = idx[stop_train_idx:]
    return train_idx, test_idx
    

def test_output(vix_reg, vix_class, eurusd_reg, eurusd_class):

    print('making directories...')
    try:
        os.mkdir('answers')
        os.chdir('answers')
    except:
        os.chdir('answers')

    attempts = os.listdir()
    try:
        attempt = max([int(attempt) for attempt in attempts if attempt.isdigit()]) + 1
    except:
        attempt = 1

    os.mkdir(str(attempt))
    os.chdir(str(attempt))

    attempt = os.getcwd()

    os.chdir('../..')


    os.mkdir('answer')
    os.chdir('answer')

    os.mkdir('VIX_1w')
    os.mkdir('EURUSDV1M_1w')

    np.savetxt('./VIX_1w/pred_reg.txt', vix_reg, fmt='%.3f')
    np.savetxt('./VIX_1w/pred_classif.txt', vix_class, fmt='%d')
    np.savetxt('./EURUSDV1M_1w/pred_reg.txt', eurusd_reg, fmt='%.3f')
    np.savetxt('./EURUSDV1M_1w/pred_classif.txt', eurusd_class, fmt='%d')

    os.chdir('..')
    print('zipping it...')
    shutil.make_archive('answer', 'zip', './answer/')
    shutil.rmtree('./answer/')
    os.chdir('../..')
    print('done!')
    print('local valid metrics:')

# def _transform_nlp_data(X, bank, nlp):
#     '''function that removes duplicates and missing values. It then tokenizes and stems properly our speeches. 
#     Finally removed rows are put back to the dataset prior to any TfIDfVectorization
#     args : 
#         - X : dataframe obtained thanks to the form_speeches_dataset function
#         - bank : either "ecb" or "fed". Argument to decide whose central bank speeches will be nlp pre-processed
#     returns :
#         - df_final : pandas Serie whose rows correspond to a tokenized and stemmed speech.'''

#     stemmer = SnowballStemmer('english')

#     speeches_bank = X[['Principal_key', f'{bank.upper()}_speeches']].drop_duplicates(f'{bank.upper()}_speeches')
#     speeches_bank = speeches_bank[speeches_bank[f'{bank.upper()}_speeches'].notnull()]
#     speeches_bank[f'{bank.upper()}_speeches'] = speeches_bank[f'{bank.upper()}_speeches'].apply(lambda element : element.split(" "))
#     speeches_bank[f'{bank.upper()}_speeches'] = speeches_bank[f'{bank.upper()}_speeches'].apply(lambda text : [re.sub(r'\W+', '', element) for element in text])
#     tokenized_tokens_bank = speeches_bank[f'{bank.upper()}_speeches'].apply(lambda text : [stemmer.stem(element) for element in text])
#     tokenized_tokens_bank_final = tokenized_tokens_bank.apply(lambda element : ' '.join(element))
    
#     df_join = X[['Principal_key', f'{bank.upper()}_speeches']].groupby([f'{bank.upper()}_speeches'])['Principal_key'].unique().reset_index().reset_index().explode('Principal_key')
#     mappers_dict = dict(zip(df_join['Principal_key'], df_join['index']))
#     df_idf = pd.DataFrame(tokenized_tokens_bank_final)
#     df_idf.rename({f'{bank.upper()}_speeches':'processed_speeches'},axis=1,inplace=True)
#     df_idf['Principal_key'] = speeches_bank['Principal_key'].values
#     df_idf['group'] = df_idf['Principal_key'].map(mappers_dict)
#     df_join = df_join.rename({'index' : 'group'}, axis=1)
#     df_final_idf = df_join.merge(df_idf.drop(['Principal_key'],axis=1), left_on='group', right_on='group')
    
#     df_null = X[X[f'{bank.upper()}_speeches'].isnull()][['Principal_key']]
#     df_null['processed_speeches'] = ''
#     df_final = pd.concat([df_final_idf.drop({f'{bank.upper()}_speeches','group'},axis=1), df_null])
    
#     df_final['Value_number'] = df_final['Principal_key'].apply(lambda element : int(element.split('-')[0]))
#     df_final['day_number'] = df_final['Principal_key'].apply(lambda element : int(element.split('-')[1]))
#     df_final = df_final.sort_values(['Value_number','day_number'])
    
#     return df_final.drop({'Principal_key','Value_number','day_number'},axis=1)

# def tf_idf_pca_reduction(X, max_df=0.5):
#     X = X.copy()
#     first_vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range=(2,2), max_features = 10_000, max_df=max_df)
#     corpus = list(X.iloc[:,0].tolist())

#     tf_idf_final = first_vectorizer.fit_transform(corpus)
    
#     components = 300
#     pca = PCA(n_components=components)
#     sc_scaler = StandardScaler()

#     idf_final_scaled = sc_scaler.fit_transform(tf_idf_final.toarray())
#     idf_300 = pca.fit_transform(idf_final_scaled)

#     return idf_300

# def get_lang_detector(nlp, name):
#     return LanguageDetector()



# def transform_non_english_none(speech,nlp):
#     language = None
#     try:
#         interesting_part = " ".join(speech.split(' ')[200:250])
#         language = nlp(interesting_part)._.language['language']
#     except:
#         pass
#     return language

# def language_transformer(X, bank,nlp):
#     X = X.copy()
#     test = X.drop_duplicates([f'{bank.upper()}_speeches'])
#     test['language'] =test[f'{bank.upper()}_speeches'].drop_duplicates().apply(lambda element : transform_non_english_none(element, nlp))
#     merger = pd.DataFrame(speeches.groupby([f'{bank.upper()}_speeches'])['Principal_key'].unique()).reset_index().reset_index().rename({'index':'group'},axis=1)
#     final_merge = test.merge(merger[['group','Principal_key']].explode('Principal_key'),left_on='Principal_key', right_on='Principal_key', how='left')
#     speech_merged = speeches.merge(merger[['group','Principal_key']].explode('Principal_key'),left_on='Principal_key', right_on='Principal_key', how='left')
#     speeches_final = speech_merged.merge(final_merge[['group','language']],left_on='group',right_on='group')
#     return speeches_final


# def form_speeches_dataset(X):
#     '''Function that transforms the raw data to a dataframe with a row per day. 
#     I.e if we have 1254 entries and 20 days per entry, it will return a dataframe with 1254 * 20 rows
#     args : 
#         - X : raw pd dataset
#     returns :
#         - speeches : transformed pd dataset  '''
#     ecb_texts = []
#     fed_texts = []
#     speech_number = []
#     day_number = []
#     for i, value in enumerate(X['speech']):
#         for j, speech in enumerate(value):
#             ecb_texts.append(speech['ECB'])
#             if len(speech['FED']) ==0:
#                 fed_texts.append(None)
#             else:
#                 fed_texts.append(speech['FED'][0])
#             speech_number.append(i)
#             day_number.append(j)

#     speeches = pd.DataFrame(ecb_texts, columns = ['ECB_speeches'])
#     speeches['FED_speeches'] = fed_texts
#     speeches['Value_number'] = speech_number
#     speeches['day_number'] = day_number
#     speeches['Principal_key'] = speeches['Value_number'].astype(str) + '-' + speeches['day_number'].astype(str)
#     return speeches




