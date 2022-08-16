import pandas as pd
import librosa 
import tqdm 
import glob

base='/home/german/Documentos/Trust/First_Impressions_Dataset/OCEAN_new_structure/'
no_unvoiced_list=glob.glob(base+'data/audios_no_unvoiced/*/*.wav')

df=pd.DataFrame()

with open (base+'experiments/audio_lists/speech_ratio_>0.5.txt','w') as file:
    for audio in tqdm.tqdm(no_unvoiced_list): 
        basename=audio.split('/')[-1]
        part=audio.split('/')[-2]    
        complete,fs=librosa.load(base+'data/audio/'+part+'/'+basename,sr=16000)
        no_unvoiced, fs= librosa.load(base+'data/audios_speech_only/'+part+'/'+basename, sr=None)
        speech=len(no_unvoiced)
        ratio=speech/len(complete)
        dict={'name':basename,
        'part':part,
        'speech_ratio':ratio}
        df=df.append(dict,ignore_index=True)

df.to_csv(base+'experiments/audio_lists/df_speech_ratio.csv')