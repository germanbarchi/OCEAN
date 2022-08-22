import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import numpy as np
import tqdm
import warnings
import seaborn as sns

def split_train_test(df):
    
    df_train=df[df['Partition']=='Train']
    df_val=df[df['Partition']=='Val']
    df_test=df[df['Partition']=='Test']

    df_train_X=df_train.iloc[:,1:-6]
    df_train_Y=df_train.iloc[:,-6:-1]

    df_val_X=df_val.iloc[:,1:-6]
    df_val_Y=df_val.iloc[:,-6:-1]

    df_test_X=df_test.iloc[:,1:-6]
    df_test_Y=df_test.iloc[:,-6:-1]

    return df_test_X, df_test_Y, df_train_X, df_train_Y, df_val_X, df_val_Y

def RandomForest(df):  
    
    X_test, Y_test, X_train, Y_train, X_val, Y_val =split_train_test(df)
    
    RF_reg=RandomForestRegressor(random_state=42) 
    
    RF_reg.fit(X_train,Y_train)
      
    predictions=RF_reg.predict(X_val.values)
      
    r2=r2_score(Y_val, predictions)    
    
    MAE=mean_absolute_error(Y_val, predictions)
    MSE=mean_squared_error(Y_val, predictions)
    RMSE=np.sqrt(mean_squared_error(Y_val, predictions))
        
    return predictions,r2,MAE,MSE,RMSE,Y_val, RF_reg

def resumen(var):
    
    big_five=['all','O','C','E','A','N']

    DF_final=pd.DataFrame()
    DF_final['index']=['r2','r','MAE','MSE','RMSE']
    DF_final=DF_final.set_index('index')    

    for dim in big_five:
        DF_final[dim]=np.array([var['r2_'+dim],np.sqrt(var['r2_'+dim]),var['MAE_'+dim],var['MSE_'+dim],var['RMSE_'+dim]])

    return DF_final   

if __name__=='__main__':
    
    df=pd.read_csv('experiments/egemaps/random_forest/default/all_audio/local_data/egemaps_all_audio_complete_set.csv',index_col=0)
    
    train_len=df[df['Partition']=='Train'].shape[0]

    n_samples=train_len
iterations=1000
metrics_list=[]

warnings.filterwarnings("ignore")

for i in tqdm.tqdm(range(iterations)):
    subset=df.sample(n=n_samples, replace=True)
    preds_all,r2_all,MAE_all,MSE_all,RMSE_all,y_test,RF_reg=RandomForest(subset)
    metrics=[r2_all,np.sqrt(r2_all),MAE_all,MSE_all,RMSE_all]
    metrics_list.append(metrics)
    
metrics_list=np.transpose(metrics_list)
df=pd.DataFrame({'r2':metrics_list[0],'r':metrics_list[1],'MAE':metrics_list[2],'MSE':metrics_list[3],'RMSE':metrics_list[4]})

df.to_csv('bootstraping.csv')