import pandas as pd
import librosa 
import tqdm 
import glob
import os 
import sys

base=os.getcwd()
no_unvoiced_list=glob.glob(base+'/data/silero/silero_speech/*/*.wav')

threshold=float(sys.argv[1])

df=pd.DataFrame()

with open (base+'/experiments/audio_lists/silero_speech_ratio_>'+str(threshold)+'.txt','w') as file:
    for audio in tqdm.tqdm(no_unvoiced_list): 
        basename=audio.split('/')[-1]
        part=audio.split('/')[-2]    
        complete,fs=librosa.load(base+'/data/audio/'+part+'/'+basename,sr=16000)
        no_unvoiced, fs= librosa.load(base+'/data/silero/silero_speech/'+part+'/'+basename, sr=None)
        speech=len(no_unvoiced)
        ratio=speech/len(complete)
        dict={'name':basename,
        'part':part,
        'speech_ratio':ratio}
        df=df.append(dict,ignore_index=True)
        if ratio>threshold:
            file.write(part+'/'+basename)

df.to_csv(base+'/experiments/audio_lists/speech_ratio_silero/df_speech_ratio_silero.csv')