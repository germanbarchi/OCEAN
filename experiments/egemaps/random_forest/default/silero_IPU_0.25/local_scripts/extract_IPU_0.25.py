import json
import librosa
import warnings
import glob
import soundfile as sf
import numpy as np
import os
import tqdm
import sys

def trim(json_path,base_path,out_path,fs,IPU):

    filename=json_path.split('/')[-1].replace('.JSON','.wav')
    part=json_path.split('/')[-2]

    y,fs=librosa.core.load(base_path+'/'+'/'.join((part,filename)), sr=fs)

    with open (json_path) as jsonfile:
        dict=json.load(jsonfile)
    
    n_segment=0 

    final=np.array([])
    segment=np.array([])

    for i in range(len(dict)-1):      
                   
        start=int(dict[i]['start'])
        end=int(dict[i]['end'])
                
        speech=y[start:end]      
   
        pause_end=int(dict[i+1]['start'])
        pause_init=int(dict[i]['end'])
                
        no_speech=y[pause_init:pause_end]
        
        speech_next=y[dict[i+1]['start']:dict[i+1]['end']] 

        pause_duration=pause_end-pause_init 
        
        if pause_duration < IPU:
            
            segment=np.append(speech,no_speech)
            final=np.append(final,segment)
            
            if i==len(dict)-2:
                
                n_segment+=1                  
                final=np.append(final,speech_next)
                
                if not os.path.exists(out_path+'/'+part):
                    os.makedirs(out_path+'/'+part)
                
                sf.write(out_path+'/'+part+'/%s_%d.wav'% (filename.replace('.wav',''),n_segment),final,fs)
        else:              

            n_segment+=1                
            final=np.append(final,speech)

            if not os.path.exists(out_path+'/'+part):
                os.makedirs(out_path+'/'+part)

            sf.write(out_path+'/'+part+'/%s_%d.wav'% (filename.replace('.wav',''),n_segment),final,fs)
            
            final=np.array([])

            if i==len(dict)-2:
                n_segment+=1 
                sf.write(out_path+'/'+part+'/%s_%d.wav' % (filename.replace('.wav',''),n_segment),speech_next,fs)

if __name__=='__main__':

    FS=16000
    audios=sys.argv[1]
    out=sys.argv[2]
    json_path=sys.argv[3]
  
    pause_threshold=0.25    
    IPU=pause_threshold*FS

    json_list=glob.glob(json_path+'/*/*.JSON')

    for json_path in tqdm.tqdm(json_list):
        trim(json_path,audios,out,FS,IPU)

    
