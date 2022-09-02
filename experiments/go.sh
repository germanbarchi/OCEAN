#!/bin/bash
######################################## Args #####################################

NOW=$(date +"%d_%m_%Y_%T")
CWD=$(pwd) # esto tiene que ser la ruta absoluta del repo
NUM_ARGS=6

if [ $# -eq 0 ] 
then   
   
   source experiments/config.sh    
   SAVE_PATH=$EXPERIMENT_PATH/results
   DATA_PATH=$CWD/$LOCAL_DATA_PATH   
   REPORT_FILE=$SAVE_PATH/${EXPERIMENT_NAME}_report_$NOW.txt   

   # Create new experiment dir
   
   if [ ! -d $EXPERIMENT_PATH ]
   then 
      experiments/new.sh $FEATURES $MODEL $PARAMS $AUDIO_TYPE      
   fi 
   
   touch $REPORT_FILE
   echo '------------------- INIT ---------------------' | tee -a $REPORT_FILE
   echo $(date): 'Load default parameters from config file: OK' | tee -a $REPORT_FILE
   echo $(date): 'Create new experiment directory: Ok' | tee -a $REPORT_FILE 
   echo $(date): 'Relative path to experiment:' $EXPERIMENT_PATH  | tee -a $REPORT_FILE   
   
   # Copy config params to results dir 

   cp experiments/config.sh $SAVE_PATH/config_${EXPERIMENT_NAME}_report_$NOW.sh 
   
   echo $(date): 'Save config params: Ok' | tee -a $REPORT_FILE
   
   # Audio lists

   AUDIO_LIST_PATH=experiments/audio_lists
   
   if [ ! -d AUDIO_LIST_PATH ]
   then
      mkdir -p $AUDIO_LIST_PATH
   fi 

elif [ $# -eq 6 ]  # ver nuevos argumentos !!! (en proceso)
then
   FEATURES=$1
   MODEL=$2
   PARAMS=$3
   AUDIO_TYPE=$4
   EXPERIMENT_PATH=$CWD/experiments/$1/$2/$3/$4
   DATA_PATH=$5
   SAVE_PATH=$6
   REPORT_FILE=$SAVE_PATH/report_$(date).txt
   touch $REPORT_FILE
   echo $(date): '------------------- INIT ---------------------' | tee -a $REPORT_FILE
   echo $(date): 'Relative path to experiment:' $EXPERIMENT_PATH | tee -a $REPORT_FILE
else 
   echo $(date): 'missing args:' $# 'provided', $(($NUM_ARGS-$#)) 'missing' | tee -a $REPORT_FILE
   echo $(date): '-usage' $0 '<features> <model> <params> <audio_type> <data_path> <save_results_path>' | tee -a $REPORT_FILE   
fi

################################### Create Dataset #####################################

# Check if audio data exists. If not, generate data at the provided path 

echo '------------------- DATA ---------------------' | tee -a $REPORT_FILE

AUDIO_PATH=$CWD/$BASE_AUDIO_PATH
NUM_FILES=10000

if [ ! -d $AUDIO_PATH ]
then
   echo $(date): 'Status: Creating data...' | tee -a $REPORT_FILE
   mkdir -p $AUDIO_PATH
   $CWD/data/scripts/create_data.sh $DATA_PATH $AUDIO_PATH

elif [ $(find $AUDIO_PATH/ -name '*.wav' | wc -l) -lt $NUM_FILES ]
then
   echo $(date): 'Status: Incomplete data... Completing ...' | tee -a $REPORT_FILE
   
   $CWD/data/scripts/create_data.sh

else
   echo $(date): 'Audio path: '$AUDIO_PATH | tee -a $REPORT_FILE

fi
   
######################################### Main ###########################################

# Audio preprocessing

echo '------------------- MAIN ---------------------' | tee -a $REPORT_FILE

PREPROCESS_FILE=$EXPERIMENT_PATH/local_scripts/$AUDIO_TYPE.sh

if [ $PREPROCESS -eq 1 ] && [ -f $PREPROCESS_FILE ]
then   
   echo "$(date): Running pre-processing" | tee -a $REPORT_FILE
    
   AUDIO_PATH=$($EXPERIMENT_PATH/local_scripts/$AUDIO_TYPE.sh $CWD $AUDIO_PATH $AUDIO_TYPE $EXPERIMENT_PATH)
    
   echo "$(date): Pre-processing: OK" | tee -a $REPORT_FILE
   echo "$(date): New audio path: $AUDIO_PATH" | tee -a $REPORT_FILE
elif [ $PREPROCESS -eq 0 ] && [ ! -z $FE_AUDIO_PATH ]
then
   AUDIO_PATH=$CWD/$FE_AUDIO_PATH
fi

# Extract features

FEATURES_FILE=${FEATURES}_$AUDIO_TYPE
FEATURES_PATH=experiments/feature_data/$FEATURES_FILE.csv

if [ $EXTRACT_FEATURES -eq 1 ]
then     

   echo "$(date): Checking existing features: $FEATURES_FILE" | tee -a $REPORT_FILE
   
   if [ -f $FEATURES_PATH ]
   then
      echo "$(date): Features exist. Loading: $FEATURES_FILE" | tee -a $REPORT_FILE
      echo "$(date): Proceeding to create dataframe" | tee -a $REPORT_FILE 
            
   else 
      echo "$(date): Features file not found" | tee -a $REPORT_FILE
      echo "$(date): Extracting features: $FEATURES_FILE" | tee -a $REPORT_FILE
      echo "$(date): Audio path= $AUDIO_PATH" | tee -a $REPORT_FILE
      
      FEATURE_EXTRACTION=code/features/$FEATURES    
      
      python3 $FEATURE_EXTRACTION.py $CWD $FEATURES_PATH $AUDIO_PATH
      
      echo "$(date): Feature extraction: OK" | tee -a $REPORT_FILE
   fi
else 
   echo "$(date): Feature extraction: Off" | tee -a $REPORT_FILE
   echo "$(date): Loading features from: $LOAD_FEATURES" | tee -a $REPORT_FILE
   FEATURES_PATH=$LOAD_FEATURES
fi 

# Create dataframe containing egemaps and personality labels


DF_NAME=${FEATURES}_$EXPERIMENT_NAME
SAVE_DF_PATH=$EXPERIMENT_PATH/local_data
CREATE_DF=$EXPERIMENT_PATH/utils/$FEATURES_FILE

DF_PATH=$CWD/$SAVE_DF_PATH/$DF_NAME.csv

python3 $CREATE_DF.py $CWD $FEATURES_PATH $SAVE_DF_PATH $AUDIO_LIST $LABELS_PATH $DF_NAME

echo "$(date): Create dataframe: OK" | tee -a $REPORT_FILE 

# Train model and save results 

echo "$(date): Training --> model: $MODEL, Params:$PARAMS" | tee -a $REPORT_FILE
MODEL=code/models/${MODEL}_$PARAMS
echo "$(date): Loading model from: $MODEL" | tee -a $REPORT_FILE

#labels= None (corresponds to 5 labels) or individual lables: 'openness','extraversion','neuroticism','conscientiousness', 'agreeableness'

MERGE_TRAIN_VAL='False'

python3 $MODEL.py $DF_PATH $MERGE_TRAIN_VAL $SAVE_PATH $EXPERIMENT_NAME $SUBSET_TRAIN $SUBSET_VAL

echo "$(date): Training and results: OK" | tee -a $REPORT_FILE

echo '------------------- OUTPUT --------------------' | tee -a $REPORT_FILE

echo "$(date): Results path: $SAVE_PATH" | tee -a $REPORT_FILE