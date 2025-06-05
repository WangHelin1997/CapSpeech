export CUDA_VISIBLE_DEVICES=0

SAVE_DIR='./agent_data' # to save processed data
CACHE_DIR='./cache' # to save dataset cache
WAV_DIR='' # downloaded capspeech-agentdb wav path
CPUS=30
N_WORKERS=8
BATCH_SIZE=64
HUB='OpenSound/CapSpeech'

python preprocess_agenttts.py \
    --hub ${HUB} \
    --save_dir ${SAVE_DIR} \
    --cache_dir ${CACHE_DIR} \
    --wav_dir ${WAV_DIR}\
    --splits train_AgentDB test_AgentDB \
    --audio_min_length 2.0 \
    --audio_max_length 20.0 

python phonemize_no_se.py \
    --save_dir ${SAVE_DIR} \
    --num_cpus ${CPUS}

python caption.py \
    --save_dir ${SAVE_DIR}

python filemaker_no_se.py \
    --save_dir ${SAVE_DIR}
