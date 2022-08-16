#!/bin/bash
CWD=$1
JSON_PATH=$CWD/data/JSON
AUDIO_PATH=$2
AUDIO_TYPE=$3
EXPERIMENT_PATH=$4
OUT_PATH=$CWD/data/audio_$AUDIO_TYPE

echo $OUT_PATH

mkdir -p $OUT_PATH

#usage basename <partition> <json_path> <out_path>
# partition: Val , Train or Test 

python $EXPERIMENT_PATH/local_scripts/remove_unvoiced.py $JSON_PATH $AUDIO_PATH $OUT_PATH 

echo $OUT_PATH