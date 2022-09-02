import pandas as pd
import sys
import os

def concat_df(df_ipu_50,labels_df):

    # Concat features and label dataframes
  
    df_ipu_50['Name']=df_ipu_50['Name'].apply(lambda x:x.replace('.wav',''))
    labels_df['audio_tag']=labels_df['audio_tag'].apply(lambda x: x.replace('.wav',''))

    for i in range(df_ipu_50.shape[0]):
        split_=df_ipu_50.loc[i,'Name'].split('_')

        if len(split_)==2:
            df_ipu_50.loc[i,'videoid']= split_[0]

        else:
            df_ipu_50.loc[i,'videoid']='_'.join(split_[:-1])

    
    final_df=pd.merge(df_ipu_50,labels_df,left_on='videoid',right_on='audio_tag')
    final_df=final_df.loc[:,~final_df.columns.isin(['Name','Part'])]
    final_df=final_df.groupby(['Partition','audio_tag']).mean().reset_index()

    return final_df

def filter_DF(DF,filter_list):

    with open (filter_list,'r') as file:
        list=file.read().splitlines()
    
    audio_names=[a.split('/')[-1] for a in list]
    filtered_df=DF[DF['audio_tag'].isin(audio_names)] 

    return filtered_df

if __name__=='__main__':

    parent_path=sys.argv[1] 
    feature_path=sys.argv[2]
    save_rel_path=sys.argv[3]
    filter_list=sys.argv[4]
    labels_path=sys.argv[5]
    df_name=sys.argv[6]

    features_abs_path=os.path.join(parent_path,feature_path)
    filter_list_abs_path=os.path.join(parent_path,filter_list)
    save_abs_path=os.path.join(parent_path,save_rel_path)
    
    if not os.path.exists(save_abs_path):
        os.makedirs(save_abs_path)

    DF_features=pd.read_csv(features_abs_path)
    DF_labels=pd.read_csv(labels_path)

    DF=concat_df(DF_features,DF_labels)  
    
    filtered_DF=filter_DF(DF,filter_list_abs_path)

    save_path=save_abs_path+'/'+df_name+'.csv'
    filtered_DF.to_csv(save_path)
    
    print(save_path)