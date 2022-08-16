import numpy as np
import librosa
import soundfile
import os 
import json 
import tqdm 
import sys 

pause_threshold=float(sys.argv[1])

base_path='/home/german/Documentos/Trust/First_Impressions_Dataset/OCEAN_new_structure/'
out_path=base_path+'data/audio_IPU_'+str(pause_threshold)+'/'

if not os.path.exists(out_path):
    os.makedirs(out_path)

with open (base_path+'experiments/audio_lists/all_audio_complete_set.txt') as filename:
    list=filename.read().splitlines()

for file in tqdm.tqdm(list): 
    
    parent_path=out_path+'/'+file.split('/')[0]
    
    if not os.path.exists(parent_path):
        os.mkdir(parent_path)
    
    with open (base_path+'data/JSON/'+file+'.JSON') as jsonfile:
        dict=json.load(jsonfile)

    y,fs=librosa.core.load(base_path+'data/audio/'+file+'.wav', sr=None)
    
    IPU=pause_threshold*fs

    final=np.array([])
    n_segment=0
    n_sub_segment=1

    if 'words' in dict:
        
        # Creo lista con las palabras reconocidas
        
        words=[x for x in dict['words'] if 'start' in x.keys()]
        
        for i in range(len(words)-1):            
                          
            start=int(words[i]['start']*fs)
            end=int(words[i]['end']*fs)
                    
            speech=y[start:end]
                    
            pause_end=int(words[i+1]['start']*fs)
            pause_init=int(words[i]['end']*fs)
                   
            no_speech=y[pause_init:pause_end]
                    
            pause_duration=pause_end-pause_init 
                
            if pause_duration < IPU:

                segment=np.append(speech,no_speech)
                final=np.append(final,segment)
                n_sub_segment+=1

            else:
                if n_sub_segment==1:
                    final=speech                    
                    
                n_segment+=1                
                soundfile.write(out_path+'%s_%d.wav'% (file,n_segment),final,fs)
                
                final=np.array([])
                n_sub_segment=1