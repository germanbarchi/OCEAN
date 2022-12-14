import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import warnings
import os
import sys


def split_train_test(df,merge_train_val,labels):
    
    df=df.drop(columns='audio_tag')
    df_X=pd.concat([df.loc[:,'F0semitoneFrom27.5Hz_sma3nz_amean':'equivalentSoundLevel_dBp'],df.loc[:,'Partition']],axis=1)
    
    if not labels=='None':

        df_Y=df.loc[:,['Partition',labels]]
    else:
        df_Y=pd.concat([df.loc[:,'extraversion':'openness'],df.loc[:,'Partition']],axis=1)

    X_test=df_X[df_X['Partition']=='Test'].drop(columns='Partition')
    Y_test=df_Y[df_Y['Partition']=='Test'].drop(columns='Partition')

    X_train=df_X[df_X['Partition']=='Train'].drop(columns='Partition')
    Y_train=df_Y[df_Y['Partition']=='Train'].drop(columns='Partition')

    X_val=df_X[df_X['Partition']=='Val'].drop(columns='Partition')
    Y_val=df_Y[df_Y['Partition']=='Val'].drop(columns='Partition')     

    if merge_train_val=='True':
        X_train=X_train.append(X_val)
        Y_train=Y_train.append(Y_val)
    
    return X_test,Y_test,X_train,Y_train,X_val,Y_val

def Linear_Regression(df,labels,merge_val_train):  
    
    X_test, Y_test, X_train, Y_train, X_val,Y_val =split_train_test(df,merge_val_train,labels)
    
    RF_reg=LinearRegression() 
    
    RF_reg.fit(X_train,Y_train)
      
    predictions=RF_reg.predict(X_val.values)
      
    r2=r2_score(Y_val, predictions)    
    
    MAE=mean_absolute_error(Y_val, predictions)
    MSE=mean_squared_error(Y_val, predictions)
    RMSE=np.sqrt(mean_squared_error(Y_val, predictions))
        
    return predictions,r2,MAE,MSE,RMSE,Y_val, RF_reg

def resumen(results_path,var):
    
    big_five=['all','O','C','E','A','N']

    DF_final=pd.DataFrame()
    DF_final['index']=['r2','r','MAE','MSE','RMSE']
    DF_final=DF_final.set_index('index')    

    for dim in big_five:
        DF_final[dim]=np.array([var['r2_'+dim],np.sqrt(var['r2_'+dim]),var['MAE_'+dim],var['MSE_'+dim],var['RMSE_'+dim]])
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    DF_final.to_csv(results_path+'/performance.csv')

if __name__=='__main__':
    
    df_path=sys.argv[1]
    merge_val_train=sys.argv[2] #False: no mergeo train y val 
    save_path=sys.argv[3]
    experiment_name=sys.argv[4]

    results_path=save_path+'/data/'+experiment_name

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    df=pd.read_csv(df_path)   
    df=df.fillna(0)
    warnings.filterwarnings('ignore')
    labels='None'
    print('Predicci??n Random Forest con labels OCEAN')
    preds_all,r2_all,MAE_all,MSE_all,RMSE_all,y_test,RF_reg=Linear_Regression(df, labels,merge_val_train)
    
    print('Predicci??n Random Forest con label OPENNESS')
    labels='openness' 
    preds_O,r2_O,MAE_O,MSE_O,RMSE_O,y_test_O,RF_reg_O=Linear_Regression(df, labels, merge_val_train)
    
    print('Predicci??n Random Forest con label CONSCIENCIOUSNESS')
    labels='conscientiousness' 
    preds_C,r2_C,MAE_C,MSE_C,RMSE_C,y_test_C,RF_reg_C=Linear_Regression(df, labels, merge_val_train)    
    
    print('Predicci??n Random Forest con label EXTRAVERSION')
    labels='extraversion' 
    preds_E,r2_E,MAE_E,MSE_E,RMSE_E,y_test_E,RF_reg_E=Linear_Regression(df, labels, merge_val_train)
    
    print('Predicci??n Random Forest con label AGREEABLENESS')
    labels='agreeableness' 
    preds_A,r2_A,MAE_A,MSE_A,RMSE_A,y_test_A,RF_reg_A=Linear_Regression(df, labels, merge_val_train)
    
    print('Predicci??n Random Forest con label NEUROTICISM')
    labels='neuroticism' 
    preds_N,r2_N,MAE_N,MSE_N,RMSE_N,y_test_N,RF_reg_N=Linear_Regression(df, labels, merge_val_train)

    print('Guardando datos en %s' % results_path)
    
    var=vars()
    resumen(results_path,var)
