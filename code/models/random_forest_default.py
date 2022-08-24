import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import warnings
import os
import sys

def subset(df,subset_train,subset_val):
    
    # Sample Train and Val subsets according to user input in config file

    print('resampling')
    df_val=resample(df[df['Partition']=='Val'],n_samples=subset_val,replace=False)
    df_train=resample(df[df['Partition']=='Train'],n_samples=subset_train,replace=False)
    
    df_=pd.concat([df_train,df_val])
    
    return df_
    
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

def RandomForest(df,labels,merge_val_train,subset_train,subset_val):  
    
    X_test, Y_test, X_train, Y_train, X_val,Y_val =split_train_test(df,merge_val_train,labels)
    
    RF_reg=RandomForestRegressor(random_state=42) 
    
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
    subset_train=int(sys.argv[5])
    subset_val=int(sys.argv[6])

    results_path=save_path+'/data/'+experiment_name

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    df=pd.read_csv(df_path)   
    df=df.fillna(0)

        # Sample Train and Val subsets according to user input in config file
    
    if (subset_train!=0) and (subset_val!=0):

        df=subset(df,subset_train,subset_val)
    
    warnings.filterwarnings('ignore')
    labels='None'
    print('Predicción Random Forest con labels OCEAN')
    preds_all,r2_all,MAE_all,MSE_all,RMSE_all,y_test,RF_reg=RandomForest(df, labels,merge_val_train,subset_train,subset_val)
    
    print('Predicción Random Forest con label OPENNESS')
    labels='openness' 
    preds_O,r2_O,MAE_O,MSE_O,RMSE_O,y_test_O,RF_reg_O=RandomForest(df, labels, merge_val_train,subset_train,subset_val)
    
    print('Predicción Random Forest con label CONSCIENCIOUSNESS')
    labels='conscientiousness' 
    preds_C,r2_C,MAE_C,MSE_C,RMSE_C,y_test_C,RF_reg_C=RandomForest(df, labels, merge_val_train,subset_train,subset_val)    
    
    print('Predicción Random Forest con label EXTRAVERSION')
    labels='extraversion' 
    preds_E,r2_E,MAE_E,MSE_E,RMSE_E,y_test_E,RF_reg_E=RandomForest(df, labels, merge_val_train,subset_train,subset_val)
    
    print('Predicción Random Forest con label AGREEABLENESS')
    labels='agreeableness' 
    preds_A,r2_A,MAE_A,MSE_A,RMSE_A,y_test_A,RF_reg_A=RandomForest(df, labels, merge_val_train,subset_train,subset_val)
    
    print('Predicción Random Forest con label NEUROTICISM')
    labels='neuroticism' 
    preds_N,r2_N,MAE_N,MSE_N,RMSE_N,y_test_N,RF_reg_N=RandomForest(df, labels, merge_val_train,subset_train,subset_val)

    print('Guardando datos en %s' % results_path)
    
    var=vars()
    resumen(results_path,var)

    importance_O=RF_reg_O.feature_importances_
    importance_C=RF_reg_C.feature_importances_
    importance_E=RF_reg_E.feature_importances_
    importance_A=RF_reg_A.feature_importances_
    importance_N=RF_reg_N.feature_importances_   

       
    # Importance

    features_list=list(df.loc[:,~df.columns.isin(['Partition','audio_tag','extraversion','conscientiousness','openness','agreeableness','neuroticism'])].columns[1:])
    
    importance_DF=pd.DataFrame({'O':importance_O,'C':importance_C,'E':importance_E,'A':importance_A,'N':importance_N})
    importance_DF['features']=features_list

    importance_DF.to_csv(results_path+'/importance.csv')

    # Busqueda de los features más importantes
    
    importance_DF=pd.melt(importance_DF,id_vars=['features'],value_vars=['O','C','E','A','N']).rename(columns={'variable':'Personality','value':'Importance'})
    percentil_95=np.percentile(importance_DF.Importance.values,95)
    relevant_features_DF=importance_DF[importance_DF['Importance']>percentil_95]

    relevant_features_DF.to_csv(results_path+'/relevant_features.csv')