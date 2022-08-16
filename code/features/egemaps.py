import sys 
import librosa
import glob
import opensmile

import pandas as pd
import tqdm
import soundfile as sf

def return_names(file_paths):
    split_path=file_paths.split('/')
    file_name=split_path[-1]
    partition=split_path[-2]

    return file_name,partition

def smile(signal,fs):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        sampling_rate=None,
        resample=False)
    df_functionals=smile.process_signal(signal,fs)
    
    return df_functionals

def concat(df,new_row):
    
    df_functionals=pd.concat([df,new_row])

    return df_functionals

if __name__=='__main__':

    #FS=16000
    df=pd.DataFrame()    
       
    parent=sys.argv[1] 
    features_path= sys.argv[2]  
    audio_path= sys.argv[3] 
    features=features_path.split('/')[-1].split('.')[0]

    print('Generando >>>%s en >>>%s' % (features,features_path))

    file_paths=glob.glob(audio_path+'/*/*.wav')
    
    for file in tqdm.tqdm(file_paths):   

        file_tag, partition=return_names(file)
                
        signal,fs=sf.read(file)
        print(file)
        if len(signal) > 0: 
            functionals=smile(signal,fs)
            
        functionals['Part']=partition
        functionals['Name']=file_tag

        if df.empty:        
            df=functionals
        else:
            df=concat(df,functionals)

    df.to_csv(features_path,index=False)
