# CapSpeech Evaluation Tools

## Get Start
Install dependicies:
```bash
conda create -n capeval python=3.9
conda activate capeval
pip install -r requirements.txt
pip install git+https://github.com/sarulab-speech/UTMOSv2.git
```

For ASR, we need:
```bash
conda install ffmpeg
```

## Evaluate pitch, monotony, speed, age, gender
RUN:
```bash
python base_eval.py
```

## Evaluate UTMOSv2
RUN:
```bash
python mos_eval.py
```

## Evaluate ASR Results
RUN:
```bash
python asr_eval.py
```

## Evaluate emotion, accent
RUN:
```bash
cd src/example/
python categorized_emotion.py
python dialect_world_dialect.py
```
Please refer to [Vox-profile](https://github.com/tiantiaf0627/vox-profile-release.git) for more evaluation tools.
