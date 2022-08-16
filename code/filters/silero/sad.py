from time import time
import torch
torch.set_num_threads(1)

from IPython.display import Audio
from pprint import pprint
import sys
import os
import glob
import tqdm 
import json
from pathlib import Path 
import pickle
import numpy as np
import soundfile

def timestamps (wav,SAMPLING_RATE):

    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
    #print(speech_timestamps) 
    
    return speech_timestamps

def save_timestamp(timestamp,ts_parent,ts_out_file):

    if not os.path.exists(ts_parent):            
      os.makedirs(ts_parent)
 
    with open(ts_out_file,'w') as f:
        json.dump(timestamp,f) 

def speech(speech_timestamps,audio_parent,save_path,sampling_rate):
    
    if not os.path.exists(audio_parent):            
      os.makedirs(audio_parent)

    save_audio(save_path, collect_chunks(speech_timestamps, wav), sampling_rate=SAMPLING_RATE) 

def non_speech(wav,speech_timestamps,audio_ns_parent,audio_ns_out_file,sr):
    
    if not os.path.exists(audio_ns_parent):            
      os.makedirs(audio_ns_parent)

    start=0

    final=np.array([])

    for i in speech_timestamps: 
      end=i['start']
      trim=wav[start:end]
      final=np.append(final,trim)
      
      start=i['end']
    
    if len(wav)-start > 0:
      final=np.append(final,wav[start:len(wav)])

    soundfile.write(audio_ns_out_file,final,sr)
    
    return final

if __name__=='__main__':

  files_path='data/audio/*/*.wav'
  audio_path='data/silero/silero_speech'
  audio_ns_path='data/silero/silero_no_speech'
  ts_path='data/silero/timestamps'

  SAMPLING_RATE = 16000     
  USE_ONNX = False # change this to True if you want to test onnx model

  model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                            model='silero_vad',
                            force_reload=True,
                            onnx=USE_ONNX)

  (get_speech_timestamps,
  save_audio,
  read_audio,
  VADIterator,
  collect_chunks) = utils 

  all_files=glob.glob(files_path)  
  
  for file in tqdm.tqdm(all_files):

      filename=file.split('/')[-1]
      part=file.split('/')[-2]
      
      out='/'.join([part,filename])

      ts_out_file=os.path.join(ts_path,out.replace('.wav','.JSON'))
      audio_out_file=os.path.join(audio_path,out)
      audio_ns_out_file=os.path.join(audio_ns_path,out)

      ts_parent=os.path.join(ts_path,part)
      audio_parent=os.path.join(audio_path,part)
      audio_ns_parent=os.path.join(audio_ns_path,part)

      if not os.path.exists(ts_out_file):
                  
            wav = read_audio(file, sampling_rate=SAMPLING_RATE)            
            
            tstamps=timestamps(wav,SAMPLING_RATE)            
            
            save_timestamp(tstamps,ts_parent,ts_out_file)   
            
            speech(tstamps,audio_parent,audio_out_file,SAMPLING_RATE)

            non_speech(wav,tstamps,audio_ns_parent,audio_ns_out_file,SAMPLING_RATE)
