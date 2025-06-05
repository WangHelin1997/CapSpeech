export CUDA_VISIBLE_DEVICES=0

SAVE_DIR='./pretrain_data' # to save processed data
CACHE_DIR='./cache' # to save dataset cache
MLS_WAV_DIR='' # downloaded mls wav path
LIBRITTSRMIX_WAV_DIR='' # downloaded librittsrmix wav path
GIGASPEECH_WAV_DIR='' # downloaded gigaspeech wav path
COMMONVOICE_WAV_DIR='' # downloaded commonvoice wav path
EMILIA_WAV_DIR='' # downloaded emilia wav path
CPUS=30
N_WORKERS=8
BATCH_SIZE=64
HUB='OpenSound/CapSpeech'

python preprocess_pretrain.py \
    --hub ${HUB} \
    --save_dir ${SAVE_DIR} \
    --cache_dir ${CACHE_DIR} \
    --libriRmix_wav_dir ${LIBRITTSRMIX_WAV_DIR}\
    --mls_wav_dir ${MLS_WAV_DIR} \
    --commonvoice_dir ${COMMONVOICE_WAV_DIR} \
    --gigaspeech_dir ${GIGASPEECH_WAV_DIR} \
    --emilia_dir ${EMILIA_WAV_DIR} \
    --splits train_PT validation_PT \
    --audio_min_length 3.0 \
    --audio_max_length 18.0 

python phonemize.py \
    --save_dir ${SAVE_DIR} \
    --num_cpus ${CPUS}

python caption.py \
    --save_dir ${SAVE_DIR}

python filemaker.py \
    --save_dir ${SAVE_DIR}

python vocab.py \
    --save_dir ${SAVE_DIR}
