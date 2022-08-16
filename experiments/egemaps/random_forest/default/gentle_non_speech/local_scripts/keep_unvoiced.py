import json
import librosa
import warnings
import glob

import soundfile
import numpy as np
import os
import tqdm
import sys

def trim (json_list_val, out_unvoiced, audios):

    for aligned in tqdm.tqdm(json_list_val):

        with open(aligned) as json_file:
            data = json.load(json_file)
            
        warnings.filterwarnings('ignore')    
        file=aligned.split('/')[-1].replace('.JSON','.wav')  
        part=aligned.split('/')[-2]
        out_path=out_unvoiced+'/'+part
        
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        
        if 'words' in data.keys():  
    
            if not os.path.exists(out_path+'/%s'% (file)):         
            
                y,fs=librosa.load(audios+'/'+part+'/'+file,sr=FS)
                
                dict={'palabras_reconocidas':[]}

                for i,word in enumerate(data['words']):
                    
                    inf=0
                    if (word.keys()>={'alignedWord','word'}):
                        word_dict={}
                        word_dict['index']=i
                        word_dict['word']=word['alignedWord']
                        word_dict['start']=word['start']
                        word_dict['end']=word['end']
                        dict['palabras_reconocidas'].append(word_dict)
                    
                    # Keep unvoiced segments

                    final_unvoiced=np.array([])

                    for word in dict['palabras_reconocidas']:
                        
                        samples_start=int(word['start']*fs)
                        samples_end=int(word['end']*fs)

                        #Keep unvoiced segments
                        
                        trim=y[inf:samples_start]
                        final_unvoiced=np.append(final_unvoiced,trim)
                        inf=samples_end                    
                    
                    soundfile.write(out_path+'/%s'% (file),final_unvoiced,FS)  

        else:
            print('filename:'+'/%s'% (file))     
            
            with open (out_unvoiced+'/faulty.txt','w') as f:
                f.write('filename:'+'/%s\n'% (file))             


if __name__=='__main__':

    FS=8000
    json_path=sys.argv[1]
    audios=sys.argv[2]
    out=sys.argv[3]

    json_list=glob.glob(json_path+'/*/*.JSON')

    trim (json_list, out, audios)