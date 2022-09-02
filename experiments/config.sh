# Features:
# * egemaps
# Model: 
# * random_forest
# Params:
# * default
# Audio type:
# * full_audio 
# * speech
# * non_speech
# * IPU_0.25
# * IPU_0.5

BASE_AUDIO_PATH=data/audio
LABELS_PATH=data/labels/new_partitions-labels.csv 

FEATURES='egemaps'

MODEL='random_forest'
#MODEL='linear_regression'

PARAMS='default'

#AUDIO_TYPE='all_audio'
#AUDIO_TYPE='IPU_0.5'
#AUDIO_TYPE='IPU_0.25'
#AUDIO_TYPE='avg_IPU_0.5'
#AUDIO_TYPE='avg_IPU_0.25'
#AUDIO_TYPE='gentle_speech'
#AUDIO_TYPE='gentle_non_speech'

#AUDIO_TYPE='silero_speech'
#AUDIO_TYPE='silero_no_speech'
#AUDIO_TYPE='silero_IPU_0.25'
#AUDIO_TYPE='silero_IPU_0.5'
#AUDIO_TYPE='silero_avg_IPU_0.25'
AUDIO_TYPE='silero_avg_IPU_0.5'

#AUDIO_TYPE='aleatorio'
#AUDIO_TYPE='aleatorio_mix_val'

LOCAL_DATA_PATH=data
EXPERIMENT_PATH=experiments/$FEATURES/$MODEL/$PARAMS/$AUDIO_TYPE

# PRE-PROCESSING

# Preprocess audio 1: On / 0: Off. Audio input dir: "BASE_AUDIO_PATH"    
# You can define "FE_AUDIO_PATH" for custom audio dir processing. 

PREPROCESS=0
FE_AUDIO_PATH=data/silero/$AUDIO_TYPE

# FEATURE EXTRACTION

# 0: No feature extraction --> You must define "LOAD_FEATURES"
# 1: Extract or use existing features from local experiment
# Features Auto-check or feature extraction.

EXTRACT_FEATURES=0 
#LOAD_FEATURES=experiments/feature_data/${FEATURES}_${AUDIO_TYPE}.csv
#LOAD_FEATURES=experiments/feature_data/new_partitions-${FEATURES}_${AUDIO_TYPE}.csv
#LOAD_FEATURES=experiments/feature_data/new_partitions-${FEATURES}_silero_IPU_0.5.csv
#LOAD_FEATURES=experiments/feature_data/${FEATURES}_silero_IPU_0.5.csv

# AUDIO SUBLIST
# Define audio file subset list

#LIST=all_audio_complete_set

#LIST=all_audio_music_0.2

#LIST=all_audio_no_music_threshold_0.2
#LIST=all_audio_no_music_threshold_0.1
#LIST=all_audio_no_music

#LIST=yamnet_no_music
#LIST=yamnet_no_music_0.2
#LIST=yamnet_no_music_0.1

#LIST=yamnet_music
#LIST=yamnet_music_0.2
#LIST=yamnet_music_0.1

#LIST=yamnet_no_music_20+speech_rate_0.5
#LIST=yamnet_no_music_20+speech_rate_0.6
#LIST=yamnet_no_music_20+speech_rate_0.7
#LIST=yamnet_no_music_20+speech_rate_0.8
#LIST=yamnet_no_music_20+speech_rate_0.9


AUDIO_LIST=experiments/audio_lists/$LIST.txt

# Define train and val subset value or set subset value to 0 to use the entire partition 

SUBSET_TRAIN=0
SUBSET_VAL=0

#EXPERIMENT_NAME=${AUDIO_TYPE}-$LIST-subset_t${SUBSET_TRAIN}_v$SUBSET_VAL

EXPERIMENT_NAME=new_partition_${AUDIO_TYPE}-$LIST