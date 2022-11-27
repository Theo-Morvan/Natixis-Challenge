import pandas as pd
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


''''
Only used for data cleaning purposes
'''

def unpack(one_speech):
    try:
        return one_speech[0]
    except:
        return np.nan



'''
Expands data from JSON format to Pandas DataFrames. Offers some testing capabilities
'''

def data_expander(
    index='vix', task='classification', test_size=0, 
    price_data=True, speech_data = True, all_data=False,
    validation = False):
    
    if index == 'vix':
        df = pd.read_json('./data/train/VIX_1w.json')
    else: 
        df = pd.read_json('./data/train/EURUSDV1M_1w.json')
    
    # Creation of a df with all the input variables 

    X = pd.DataFrame(index=df.index)

    prices = pd.DataFrame(
            df['stock'].to_list(),
            columns=['price' + str(i) for i in range(1,21)]
            )

    speeches = pd.DataFrame(df['speech'].values.tolist())
    speech_cols = pd.DataFrame(index=speeches.index)

    for i in range(len(speeches.columns)):
        daily = pd.DataFrame(speeches[i].values.tolist())
        daily.columns = [f'ecb{i+1}', f'fed{i+1}']
        speech_cols = speech_cols.join(daily)

    speech_cols = speech_cols.applymap(unpack)

    if price_data:
        X = X.join(prices)
    if speech_data:
        X = X.join(speech_cols)
    
    # Target variable

    if task == 'classification':
        y = df['target_classif']
    else:
        y = df['target_reg']

    # Return entire frame

    if all_data:
        return X, df['target_classif'], df['target_reg']

    # Validation data, stay away

    X_train, X_validation, y_train, y_validation = train_test_split(
            X, y, test_size=0.2, random_state=1)

    if validation:
        return X_train, X_validation, y_train, y_validation

    # Train-test split
    
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=test_size, random_state=1)
        return X_train, X_test, y_train, y_test
    else:
        return X_train, y_train



'''
Gets accuracy and rmse for all tasks
'''

def metrics(
    vixclassifier=None, vixregressor=None,
    eurclassifier=None, eurregressor=None, 
    prices=True, speeches=True,
    test_split=0.2, valid=False):

    if eurclassifier == None:
        eurclassifier = vixclassifier
    if eurregressor == None:
        eurregressor = vixregressor

    if valid:
        test_split = 0

    accuracy = {}
    rmse = {}

    if vixclassifier:

        X_train, X_test, y_train, y_test = data_expander(
            task='classification', index='vix',
            speech_data=speeches, price_data=prices,
            test_size=test_split, validation=valid)

        vixclassifier.fit(X_train, y_train)
        predictions = vixclassifier.predict(X_test)
        accuracy['vix'] = accuracy_score(predictions, y_test)

    if vixregressor:

        X_train, X_test, y_train, y_test = data_expander(
            task='regression', index='vix',
            speech_data=speeches, price_data=prices,
            test_size=test_split, validation=valid)

        vixregressor.fit(X_train, y_train)
        predictions = vixregressor.predict(X_test)
        rmse['vix'] = mean_squared_error(predictions, y_test, squared=False)

    if eurclassifier:

        X_train, X_test, y_train, y_test = data_expander(
            task='classification', index='eurusd',
            speech_data=speeches, price_data=prices,
            test_size=test_split, validation=valid)

        eurclassifier.fit(X_train, y_train)
        predictions = eurclassifier.predict(X_test)
        accuracy['eurusd'] = accuracy_score(predictions, y_test)

    if eurregressor:

        X_train, X_test, y_train, y_test = data_expander(
            task='regression', index='eurusd',
            speech_data=speeches, price_data=prices,
            test_size=test_split, validation=valid)

        eurregressor.fit(X_train, y_train)
        predictions = eurregressor.predict(X_test)
        rmse['eurusd'] = mean_squared_error(predictions, y_test, squared=False)

    return pd.DataFrame([accuracy, rmse], index = ['Accuracy', 'RMSE'])



'''
Unpacks test data in a similar fashion to data_expander
Only used in backend
'''

def test_data(index='vix', price_data=True, speech_data=True):
    
    if index == 'vix':
        df = pd.read_json('./data/dev/VIX_1w.json')
    else: 
        df = pd.read_json('./data/dev/EURUSDV1M_1w.json')
    
    # Creation of a df with all the input variables 

    X = pd.DataFrame(index=df.index)

    prices = pd.DataFrame(
            df['stock'].to_list(),
            columns=['price' + str(i) for i in range(1,21)]
            )

    speeches = pd.DataFrame(df['speech'].values.tolist())
    speech_cols = pd.DataFrame(index=speeches.index)

    for i in range(len(speeches.columns)):
        daily = pd.DataFrame(speeches[i].values.tolist())
        daily.columns = [f'ecb{i+1}', f'fed{i+1}']
        speech_cols = speech_cols.join(daily)

    speech_cols = speech_cols.applymap(unpack)

    if price_data:
        X = X.join(prices)
    if speech_data:
        X = X.join(speech_cols)

    return X



'''
Writes predictions using test data
Only used in backend
'''

def answers(vixclassifier, vixregressor, eurclassifier=None, eurregressor=None, prices=True, speeches=True):
    
    if eurregressor==None:
        eurregressor = vixregressor
    if eurclassifier==None:
        eurclassifier = vixclassifier

    X, y_class, y_reg = data_expander(
        all_data=True, index='vix',
        speech_data=speeches, price_data=prices)

    X_test = test_data(
        index='vix',
        price_data=prices,
        speech_data=speeches)
    
    vixclassifier.fit(X, y_class)
    vixregressor.fit(X, y_reg)
    vix_class = vixclassifier.predict(X_test)
    vix_reg = vixregressor.predict(X_test)

    X, y_class, y_reg = data_expander(
        all_data=True, index='eurusd', 
        speech_data=speeches, price_data=prices)

    X_test = test_data(
        index='eurusd',
        price_data=prices,
        speech_data=speeches)
    
    eurclassifier.fit(X, y_class)
    eurregressor.fit(X, y_reg)
    eurusd_class = eurclassifier.predict(X_test)
    eurusd_reg = eurregressor.predict(X_test)

    return vix_class, vix_reg, eurusd_class, eurusd_reg



'''
Writes zipped output ready for submission
'''

def test_output(vixclassifier, vixregressor, eurclassifier=None, eurregressor=None, price_data=True, speech_data=True):

    if eurregressor==None:
        eurregressor = vixregressor
    if eurclassifier==None:
        eurclassifier = vixclassifier

    print('getting data...')
    print('training models...')

    vix_class, vix_reg, eurusd_class, eurusd_reg = answers(
        vixclassifier, vixregressor, eurclassifier, eurregressor,
        prices=price_data, speeches=speech_data)

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

    print('writing metrics...')
    test_metrics = metrics(
        vixclassifier, vixregressor,
        prices=price_data, speeches=speech_data,
        valid=True)

    os.chdir(attempt)

    test_metrics.to_csv('metrics.csv')

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
    print(test_metrics)