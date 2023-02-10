#!/bin/python
#This file transforms a tiff image to a dataframe

import os
import logging
import glob 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

def set_environment(global_vars: dict) -> int:
    ''' Sets all the variables for the environments into a params variables

    params:
    -------
        global_vars :   dict : all the variables will be stored in this dictionnary

    returns:
        0           : standard linux return's value without an error

    '''

    global_vars['inputdir']='/inputdir'
    global_vars['outputdir']='/outputdir'
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    global_vars['logger'] = logging.getLogger(__name__)
    global_vars['cal_ndvi'] = True
    global_vars['batch_size'] = 10
    global_vars['ini_estimators'] = 20
    global_vars['add_estimators'] = 20
    global_vars['shuffle'] = True
    global_vars['modelname'] = 'rf.sav.gzip'


    
    return 0



desc = """
    Training a random forest model recursively  
    inputs : /inputdir : full paths of the img with class names and img csv files
        /outputdir : calibraited models   
"""

def split_chunck(global_var: dict, imgs: list) -> list: 
    '''splites a list into equally spaced trucnks
    '''
    batch_size = global_var['batch_size']
    n  = len(imgs)

    return [imgs[i:i+batch_size] for i in range(0,n,batch_size)]


def df_bundle(global_var: dict, imgs: list) -> pd.DataFrame:
    '''Creates a pandas dataframe by appending verticaly csv dataframes
    '''

    shuffle = global_var['shuffle']
    logger = global_var['logger']

    dfs = []
    for img in imgs:
        dfs.append(pd.read_csv(img, 
            compression={'method': 'gzip', 'compresslevel': 6}))

    df_result = pd.concat(dfs,
          ignore_index=True)

    if shuffle : 
        df_result = df_result.sample(frac=1).reset_index(drop=True)

    logger.debug('df_result_size {0}'.format(df_result.shape))
    logger.debug('df_result {0}'.format(df_result.head()))

    return df_result

def do_training(globar_var: dict, batch_imgs: list) -> object:
    '''creates the training set from an img csv and a mask csv
       from a list of batch images
    '''
    logger = global_var['logger']
    ini_estimators=global_var['ini_estimators'] 
    add_estimators= global_var['add_estimators']   

    n_estimators = ini_estimators

    clf = RandomForestClassifier(n_estimators=n_estimators, 
            warm_start=True,
            max_depth = 10, 
            n_jobs = -1,
            max_samples = 0.2,
            bootstrap = True)

    logger.debug("len of batch_imgs {0}".format(len(batch_imgs)))
    n_epochs = len(batch_imgs)

    for i,obatch in enumerate(batch_imgs):
        logger.info('remaining epochs rf {0}'.format(n_epochs-i))

        df_train = df_bundle(global_var, obatch)
        X =df_train[['R','G','B','N','H','ndvi']].values
        Y =df_train[['C']].values.reshape(-1)
        clf.fit(X, Y)
        
        n_estimators += add_estimators
        clf.set_params(n_estimators=n_estimators)

    return clf

def do_save(global_var: dict, model: object) -> int:
    '''save and compress a model
    '''
    outputdir=global_var['outputdir']
    modelname = global_var['modelname']
    logger = global_var['logger']

    logger.info('saving model to file {0}'.format(modelname))

    joblib.dump(model, os.path.join(outputdir,modelname), compress=('gzip',3))

    return 0

def do_load(global_var: dict) -> object:
    ''' load a trained model from the outputdir
    '''
    outputdir=global_var['outputdir']
    modelname = global_var['modelname']
    logger = global_var['logger']

    logger.info('loading model from file {0}'.format(modelname))

    result = joblib.load(os.path.join(outputdir,modelname))

    return result

def do_batch(global_var: dict) -> list:
    '''Creates the batch lists of chuncks
    '''
    inputdir = global_var['inputdir']
    logger = global_var['logger']

    imgs_files = glob.glob(os.path.join(inputdir,'**','IMG_*.csv.gzip'), recursive=True)

    logger.debug(imgs_files)

    batch_imgs = split_chunck(global_var, imgs_files)

    return batch_imgs


def do_predict(globar_var: dict, model: object, batch_imgs: list) -> int: 
    ''' Test error on test set
    '''
    logger = global_var['logger']
    df_test = df_bundle(global_var, batch_imgs)
    X = df_test[['R','G','B','N','H','ndvi']].values
    Y = df_test[['C']].values.reshape(-1)

    logger.info('Score of randon Forest {0}'.format(model.score(X, Y)))

    return 0


if __name__ == '__main__':
    
    global_var = {}
    set_environment(global_var)
    logger = global_var['logger']
    logger.info(desc)

    logger.info('global_var {0}'.format(global_var))

    logger.info('training ... ')

    batch_imgs = do_batch(global_var)
    tr_batch_imgs = batch_imgs[0:1]
    ts_batch_imgs = batch_imgs[-1]

    #model_rf=do_training(global_var, tr_batch_imgs)

    #do_save(global_var, model_rf)

    model_rf = do_load(global_var)
    
    do_predict(global_var, model_rf, ts_batch_imgs)




    




