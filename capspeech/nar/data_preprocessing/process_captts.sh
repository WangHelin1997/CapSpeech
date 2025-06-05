export CUDA_VISIBLE_DEVICES=0

SAVE_DIR='./captts_data' # to save processed data
CACHE_DIR='./cache' # to save dataset cache
LIBRITTSR_WAV_DIR='' # downloaded libritts-r wav path
OTHER_WAV_DIR='' # downloaded other wav path
CPUS=30
N_WORKERS=8
BATCH_SIZE=64
HUB='OpenSound/CapSpeech'

python preprocess_captts.py \
    --hub ${HUB} \
    --save_dir ${SAVE_DIR} \
    --cache_dir ${CACHE_DIR} \
    --libriR_wav_dir ${LIBRITTSR_WAV_DIR}\
    --other_wav_dir ${OTHER_WAV_DIR} \
    --splits train_SFT_CapTTS validation_SFT_CapTTS \
    --audio_min_length 3.0 \
    --audio_max_length 18.0 

python phonemize_no_se.py \
    --save_dir ${SAVE_DIR} \
    --num_cpus ${CPUS}

python caption.py \
    --save_dir ${SAVE_DIR}

python filemaker_no_se.py \
    --save_dir ${SAVE_DIR}
