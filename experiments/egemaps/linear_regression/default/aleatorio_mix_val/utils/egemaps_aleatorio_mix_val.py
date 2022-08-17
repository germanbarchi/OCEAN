import pandas as pd
import sys
import os

def concat_df(DF,labels_df):

    # Concat features and label dataframes
  
    DF['Name']=DF['Name'].apply(lambda x:x.replace('.wav',''))
    labels_df['audio_tag']=labels_df['audio_tag'].apply(lambda x: x.replace('.mp4',''))
    
    #mix only validation labels 

    labels_train=labels_df[labels_df['Partition']=='Train'].copy()
    labels_val=labels_df[labels_df['Partition']=='Val'].copy()
    labels_test=labels_df[labels_df['Partition']=='Test'].copy()

    labels_val.loc[:,'extraversion':'openness']=labels_val.loc[:,'extraversion':'openness'].sample(frac=1).values

    labels_df=pd.concat([labels_train,labels_val,labels_test])

    final_df=pd.merge(DF,labels_df,left_on='Name',right_on='audio_tag')
    final_df=final_df.loc[:,~final_df.columns.isin(['Name','Part'])]

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
    name=sys.argv[6]

    features_abs_path=os.path.join(parent_path,feature_path)
    filter_list_abs_path=os.path.join(parent_path,filter_list)
    save_abs_path=os.path.join(parent_path,save_rel_path)
       
    if not os.path.exists(save_abs_path):
        os.makedirs(save_abs_path)

    features=pd.read_csv(features_abs_path)
    labels_df=pd.read_csv(labels_path)

    DF=concat_df(features,labels_df)  
    
    filtered_DF=filter_DF(DF,filter_list_abs_path)

    save_path=save_abs_path+'/'+name+'.csv'
    filtered_DF.to_csv(save_path)
    
    print(save_path)