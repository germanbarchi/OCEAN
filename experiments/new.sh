#!/bin/bash

FEATURES=$1
MODEL=$2
PARAMS=$3
AUDIO_TYPE=$4 
NEW_METHOD='new_method'

EXPERIMENT_PATH=experiments/$FEATURES/$MODEL/$PARAMS/$AUDIO_TYPE

FEATURES_FILE=${FEATURES}_$AUDIO_TYPE
DF_NAME=$FEATURES_FILE+labels_DF
CREATE_DF=$EXPERIMENT_PATH/utils/$DF_NAME

if [ ! -d $EXPERIMENT_PATH ]
then     
   mkdir -p $EXPERIMENT_PATH/local_data
   mkdir $EXPERIMENT_PATH/local_files
   mkdir $EXPERIMENT_PATH/local_scripts
   touch $EXPERIMENT_PATH/local_scripts/$AUDIO_TYPE.sh
   touch $EXPERIMENT_PATH/local_scripts/$NEW_METHOD.py
   mkdir $EXPERIMENT_PATH/results 
   mkdir $EXPERIMENT_PATH/utils/
   touch $EXPERIMENT_PATH/utils/$FEATURES_FILE.py
   echo "New experiment dir created at: $EXPERIMENT_PATH" | tee -a $REPORT_FILE
fi
