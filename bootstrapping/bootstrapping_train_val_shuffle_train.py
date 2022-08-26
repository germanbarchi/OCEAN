import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.utils import resample
import numpy as np
import tqdm
import warnings
import seaborn as sns

def split_train_test(df_part):
    
    df_X=df_part.iloc[:,1:-6]
    df_Y=df_part.iloc[:,-6:-1]

    return df_X, df_Y

def RandomForest(train,val):  
    
    X_train, Y_train =split_train_test(train)
    X_val, Y_val= split_train_test(val)
    
    #Shufle train labels
    Y_train=Y_train.sample(frac=1)

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
    
    df=pd.read_csv('egemaps_all_audio_complete_set.csv',index_col=0)
    
    # subsample dataframe. Return only train and val partitions
    df_train_val=df[df['Partition'].isin(['Train','Val'])]
    
    train_len=df[df['Partition']=='Train'].shape[0]

    n_samples=train_len
    
    iterations=1000
    metrics_list=[]

    warnings.filterwarnings("ignore")

    for i in tqdm.tqdm(range(iterations)):       
        
        # resample train partition. 
        
        train = resample(df_train_val, replace=False, n_samples=train_len,random_state=42+i)
        test = df_train_val[~df_train_val.index.isin(train.index)]

        preds_all,r2_all,MAE_all,MSE_all,RMSE_all,y_test,RF_reg=RandomForest(train,test)
        metrics=[r2_all,np.sqrt(r2_all),MAE_all,MSE_all,RMSE_all]
        metrics_list.append(metrics)
        
    metrics_list=np.transpose(metrics_list)
    df=pd.DataFrame({'r2':metrics_list[0],'r':metrics_list[1],'MAE':metrics_list[2],'MSE':metrics_list[3],'RMSE':metrics_list[4]})

    df.to_csv('bootstraping_suffle_train_no_replacement.csv')
