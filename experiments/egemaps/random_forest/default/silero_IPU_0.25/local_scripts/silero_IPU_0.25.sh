CWD=$1 
AUDIO_PATH=$2 
AUDIO_TYPE=$3 
EXPERIMENT_PATH=$4

files_path=$AUDIO_PATH
ts_path=$CWD/data/silero/timestamps
audio_out_path=$CWD/data/silero/silero_IPU_0.25

python $EXPERIMENT_PATH/local_scripts/extract_IPU_0.25.py $files_path $audio_out_path $ts_path 

echo $audio_out_path 