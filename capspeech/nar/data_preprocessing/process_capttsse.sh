export CUDA_VISIBLE_DEVICES=0

SAVE_DIR='./capttsse_data' # to save processed data
CACHE_DIR='./cache' # to save dataset cache
LIBRITTSRMIX_WAV_DIR='' # downloaded librittsrmix wav path
CPUS=30
N_WORKERS=8
BATCH_SIZE=64
HUB='OpenSound/CapSpeech'

python preprocess_capttsse.py \
    --hub ${HUB} \
    --save_dir ${SAVE_DIR} \
    --cache_dir ${CACHE_DIR} \
    --libriRmix_wav_dir ${LIBRITTSRMIX_WAV_DIR}\
    --splits train_SEDB \
    --audio_min_length 3.0 \
    --audio_max_length 18.0 

python phonemize.py \
    --save_dir ${SAVE_DIR} \
    --num_cpus ${CPUS}

python caption.py \
    --save_dir ${SAVE_DIR}

python filemaker.py \
    --save_dir ${SAVE_DIR}
