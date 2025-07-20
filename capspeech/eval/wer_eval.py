
import librosa
import json
from jiwer import wer as calculate_wer
from jiwer import cer as calculate_cer
from whisper.normalizers import EnglishTextNormalizer
import whisper
import torch
import pandas as pd
import os
import librosa
import soundfile as sf
import glob
from tqdm import tqdm
import numpy as np

normalizer = EnglishTextNormalizer()
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("large-v3-turbo", device=device)

def asr(wav_path):
    result = whisper_model.transcribe(wav_path)
    pred = result['text'].strip()
    pred = normalizer(pred)
    return pred


# cal
df = pd.read_csv("CapSpeech-test.csv")
test_dir = "your_captts_results"
resample_dir = "your_captts_results_resampled16k"
save_df = "captts_result.csv"

os.makedirs(resample_dir,exist_ok=True)
# resample

audios = glob.glob(os.path.join(test_dir, "**/*.wav"), recursive=True)
for audio in tqdm(audios):
    y, sr = librosa.load(audio, sr=None)
    target_sr = 16000
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    target_path = os.path.join(resample_dir, audio.split("/")[-1])
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    sf.write(target_path, y_resampled, target_sr)

asr_preds, wer_preds, cer_preds = [],[],[]

for savename, gt_text in tqdm(zip(df["audio_path"], df["text"])):
    savename = savename.split('/')[-1]
    audiopath = os.path.join(resample_dir, savename)
    gt_text = normalizer(gt_text.strip())
    pred_asr = asr(audiopath)
    wer = round(calculate_wer(gt_text, pred_asr), 3)
    cer = round(calculate_cer(gt_text, pred_asr), 3)
    asr_preds.append(pred_asr)
    wer_preds.append(wer)
    cer_preds.append(cer)

df["asr_preds"] = asr_preds
df["wer_preds"] = wer_preds
df["cer_preds"] = cer_preds

df.to_csv(save_df, index=False)
