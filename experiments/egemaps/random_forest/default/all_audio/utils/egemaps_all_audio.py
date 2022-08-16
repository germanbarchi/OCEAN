import pandas as pd
import sys
import os

def concat_df(DF_features,DF_labels):

    # Concat features and label dataframes

    DF_features=DF_features.rename(columns={'Name':'audio_tag'})
    DF_features=DF_features.set_index('audio_tag')

    DF_labels['audio_tag']=DF_labels['audio_tag'].apply(lambda x: x.replace('.mp4','.wav'))
    DF_labels=DF_labels.set_index('audio_tag')

    DF=pd.merge(DF_features,DF_labels,how='outer',left_index=True,right_index=True).drop(columns='Part')
    DF=DF.reset_index()

    return DF

def filter_DF(DF,filter_list):

    with open (filter_list,'r') as file:
        list=file.read().splitlines()
    
    audio_names=[a.split('/')[1]+'.wav' for a in list]
    filtered_df=DF[DF['audio_tag'].isin(audio_names)] 

    return filtered_df

if __name__=='__main__':

    parent_path=sys.argv[1] 
    feature_path=sys.argv[2]
    parent=sys.argv[3]
    filter_list=sys.argv[4]
    labels_path=sys.argv[5]
    df_name=sys.argv[6]

    features_abs_path=os.path.join(parent_path,feature_path)
    filter_list_abs_path=os.path.join(parent_path,filter_list)

    if not os.path.exists(parent):
        os.mkdir(parent)

    DF_features=pd.read_csv(features_abs_path)
    DF_labels=pd.read_csv(labels_path)

    DF=concat_df(DF_features,DF_labels)  
    
    filtered_DF=filter_DF(DF,filter_list_abs_path)

    save_path=parent+'/'+df_name+'.csv'
    filtered_DF.to_csv(save_path)
    
    print(save_path)