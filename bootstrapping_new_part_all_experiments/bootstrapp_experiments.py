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
import glob
import tqdm

def split_X_Y(df_part):
    
    df_X=df_part.iloc[:,1:-6]
    df_Y=df_part.iloc[:,-6:-1]

    return df_X, df_Y

def RandomForest(train,val):  
    
    X_train, Y_train =split_X_Y(train)
    X_val, Y_val= split_X_Y(val)
    
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

def bootstrap(df_path,df_new_partitions):
    
    metrics_list=[]
    
    df=pd.read_csv(df_path,index_col=0)
    df_name=df_path.split('/')[-1]

    df_train=df_new_partitions[df_new_partitions['part_new']=='Train'].drop_duplicates(subset='basename')
    df_val=df_new_partitions[df_new_partitions['part_new']=='Val'].drop_duplicates(subset='basename')

    warnings.filterwarnings("ignore")

    for i in tqdm.tqdm(range(iterations)):       
        
        # resample train partition. 
        
        train = resample(df_train, replace=True, n_samples=n_train)
        val = resample(df_val, replace=True, n_samples=n_val)

        train_all=df[df['audio_tag'].str.contains('|'.join(train.basename.values))]
        val_all=df[df['audio_tag'].str.contains('|'.join(val.basename.values))]

        preds_all,r2_all,MAE_all,MSE_all,RMSE_all,y_test,RF_reg=RandomForest(train_all,val_all)
        metrics=[r2_all,np.sqrt(r2_all),MAE_all,MSE_all,RMSE_all]
        metrics_list.append(metrics)
        
    metrics_list=np.transpose(metrics_list)
    df=pd.DataFrame({'r2':metrics_list[0],'r':metrics_list[1],'MAE':metrics_list[2],'MSE':metrics_list[3],'RMSE':metrics_list[4]})

    df.to_csv('bootstrapping_'+df_name)

if __name__=='__main__':
    
    df_paths=glob.glob('./dataframes/*new_partition*')
    df_new_partitions=pd.read_csv('./dataset_new_partitions.csv')
    
    n_train=300
    n_val=60    
    iterations=1000

    for df_path in tqdm.tqdm(df_paths):      

        bootstrap(df_path,df_new_partitions)
    
   
