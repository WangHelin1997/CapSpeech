# CapSpeech-NAR


## Preprocess Data

You can use `data/process.sh` or run them step by step.

1. Prepare json files. Run:
```bash
SAVE_DIR='./capspeech' # to save processed data
CACHE_DIR='./cache' # to save dataset cache
MLS_WAV_DIR='' # downloaded mls wav path
LIBRITTSRMIX_WAV_DIR='' # downloaded librittsrmix wav path
GIGASPEECH_WAV_DIR='' # downloaded gigaspeech wav path
COMMONVOICE_WAV_DIR='' # downloaded commonvoice wav path
EMILIA_WAV_DIR='' # downloaded emilia wav path
CPUS=30
N_WORKERS=8
BATCH_SIZE=64
python preprocess.py \
    --save_dir ${SAVE_DIR} \
    --cache_dir ${CACHE_DIR} \
    --libriRmix_wav_dir ${LIBRITTSRMIX_WAV_DIR}\
    --mls_wav_dir ${MLS_WAV_DIR} \
    --commonvoice_dir ${COMMONVOICE_WAV_DIR} \
    --gigaspeech_dir ${GIGASPEECH_WAV_DIR} \
    --emilia_dir ${EMILIA_WAV_DIR} \
    --splits train val \
    --audio_min_length 3.0 \
    --audio_max_length 18.0 
```
Notes: `SAVE_DIR` is the path to save processed data; `CACHE_DIR` is the path to save downloaded huggingface data; `MLS_WAV_DIR` is the path of downloaded MLS English-version wav path, it should contain something like `mls_english/test/audio/10226/10111/10226_10111_000001.flac`; `COMMONVOICE_WAV_DIR` is the path of downloaded Commonvoice English-version wav path, it should contain something like `commonvoice/common_voice_en_20233751.wav`; `GIGASPEECH_WAV_DIR` is the path of downloaded GigaSpeech wav path, it should contain something like `gigaspeech/AUD0000000468_S0000654.wav`; `LIBRITTSRMIX_WAV_DIR` is the path of downloaded LibriTTS-r Mix wav path, it should contain something like `LibriTTS_R/test-clean/1089/134686/1089_134686_000001_000001_01.wav`; `EMILIA_WAV_DIR` is the path of downloaded Emilia wav path, it should contain something like `EN_B00020_S00165_W000096.mp3`.

You will get a `jsons` folder with `.json` files like this:
```
[
    {
        "segment_id": "1089_134686_000001_000001_01",
        "audio_path": "/data/capspeech-data/librittsr-mix/LibriTTS_R/test-clean/1089/134686/1089_134686_000001_000001_01.wav",
        "text": "<train_whistling> he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled <B_start> out in thick peppered flour fattened sauce stuff it into you his belly counselled him <B_end>",
        "caption": "A middle-aged male's speech is characterized by a steady, slightly somber tone, with his voice carrying a moderately low pitch. His speech pace is moderate, neither too quick nor too slow, lending an air of calm and measured thoughtfulness to his delivery.",
        "duration": 12.79125,
        "source": "libritts-r"
    },
    ...
]
```

2. Phonemize. Run:
```bash
SAVE_DIR='./capspeech'
CPUS=30
python phonemize.py \
    --save_dir ${SAVE_DIR} \
    --num_cpus ${CPUS}
```

You will get a `g2p` folder with `.txt` files.

3. Caption with T5 embeddings. Run:
```bash
SAVE_DIR='./capspeech'
python caption.py \
    --save_dir ${SAVE_DIR}
```

You will get a `t5` folder with `.npz` files.

4. Make manifests. Run:
```bash
SAVE_DIR='./capspeech'
python filemaker.py \
    --save_dir ${SAVE_DIR}
```

You will get a `manifest` folder with `.txt` files like this:
```
1995_1826_000016_000004_01	playing_accordion
1995_1826_000016_000007_01	underwater_bubbling
1995_1826_000016_000008_01	telephone
1995_1826_000016_000009_01	eletric_blender_running
1995_1826_000016_000010_01	harmonica
```

5. Make vocab. Run:
```bash
SAVE_DIR='./capspeech'
python vocab.py \
    --save_dir ${SAVE_DIR}
```

You will get a `vocab.txt` file.

📝 **Note:** We provided the following scripts to process our data. Make sure to change to your path.

1. Preprocess pretraining data:
```bash
bash data_preprocessing/process_pretrain.sh
```
2. Preprocess CapTTS, EmoCapTTS and AccCapTTS data:
```bash
bash data_preprocessing/process_captts.sh
```
3. Preprocess CapTTS-SE data:
```bash
bash data_preprocessing/process_capttsse.sh
```
4. Preprocess AgentTTS data:
```bash
bash data_preprocessing/process_agenttts.sh
```

## Pretrain
```bash
accelerate launch train.py --config-name "./configs/pretrain.yaml"
```

## Finetune on CapTTS
```bash
accelerate launch finetune.py --config-name "./configs/finetune_captts.yaml" --pretrained-ckpt "YOUR_MODEL_PATH"
```

## Finetune on EmoCapTTS
```bash
accelerate launch finetune.py --config-name "./configs/finetune_emocaptts.yaml" --pretrained-ckpt "YOUR_MODEL_PATH"
```

## Finetune on AccCapTTS
```bash
accelerate launch finetune.py --config-name "./configs/finetune_acccaptts.yaml" --pretrained-ckpt "YOUR_MODEL_PATH"
```

## Finetune on CapTTS-SE
```bash
accelerate launch finetune.py --config-name "./configs/finetune_capttsse.yaml" --pretrained-ckpt "YOUR_MODEL_PATH"
```

## Finetune on AgentTTS
```bash
accelerate launch finetune.py --config-name "./configs/finetune_agenttts.yaml" --pretrained-ckpt "YOUR_MODEL_PATH"
```

## Train a duration predictor
```bash
python duration_predictor.py
```

