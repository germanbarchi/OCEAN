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
        
    return r2,MAE,MSE,RMSE

def bootstrap(df_path):
    
    metrics_list=[]
    
    df=pd.read_csv(df_path,index_col=0)
    df = df.dropna() 

    for i in range(df.shape[0]):
        df.loc[i,'basename']=df.loc[i,'audio_tag'].split('.')[0]

    df_train=df[df['Partition']=='Train'].drop_duplicates(subset='basename')
    df_val=df[df['Partition']=='Val'].drop_duplicates(subset='basename')

    warnings.filterwarnings("ignore")

    for i in tqdm.tqdm(range(iterations)):       
        
        # resample train partition. 
        
        train = resample(df_train, replace=True, n_samples=n_train)
        val = resample(df_val, replace=True, n_samples=n_val)

        train_all=df[df['audio_tag'].str.contains('|'.join(train.basename.values))]
        val_all=df[df['audio_tag'].str.contains('|'.join(val.basename.values))]

        r2_all,MAE_all,MSE_all,RMSE_all=RandomForest(train_all,val_all)
        metrics=[r2_all,np.sqrt(r2_all),MAE_all,MSE_all,RMSE_all]
        metrics_list.append(metrics)
        
    metrics_list=np.transpose(metrics_list)
    df=pd.DataFrame({'r2':metrics_list[0],'r':metrics_list[1],'MAE':metrics_list[2],'MSE':metrics_list[3],'RMSE':metrics_list[4]})

    return df

if __name__=='__main__':
    
    df_paths=glob.glob('./dataframes/*new_partition*')
        
    n_train=300
    n_val=60    
    iterations=1000

    for df_path in tqdm.tqdm(df_paths):      
        
        df_name=df_path.split('/')[-1]
        
        df_=bootstrap(df_path)
    
        df_.to_csv('bootstrapping_'+df_name)
