import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from huggingface_hub import snapshot_download

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from capspeech.nar import bigvgan
import librosa
from capspeech.nar.utils import make_pad_mask
from capspeech.nar.model.modules import MelSpec
from capspeech.nar.network.crossdit import CrossDiT
from capspeech.nar.inference import sample
from capspeech.nar.utils import load_yaml_with_includes
import soundfile as sf
from transformers import T5EncoderModel, AutoTokenizer
from g2p_en import G2p
import laion_clap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import time

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

valid_symbols = [
      'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
      'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
      'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
      'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
      'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
      'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
      'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', '<BLK>', ',', '.', '!', '?', 
      '<B_start>', '<B_end>', '<I_start>', '<I_end>'
    ]

def encode(text, text_tokenizer):
    if '<B_start>' in text:
        assert '<B_end>' in text, text
        text = text.split(">", 1)[1].strip() # remove the audio label
        seg1 = text.split("<B_start>")[0]
        seg2 = text.split("<B_start>")[1].split("<B_end>")[0]
        seg3 = text.split("<B_end>")[1]
        phn1 = text_tokenizer(seg1)
        if len(phn1) > 0:
            phn1.append(" ")
        phn1.append("<B_start>")
        phn1.append(" ")
        phn2 = text_tokenizer(seg2)
        if len(phn2) > 0:
            phn2.append(" ")
        phn2.append("<B_end>")
        phn3 = text_tokenizer(seg3)
        if len(phn3) > 0:
            phn2.append(" ")
        phn = [*phn1,*phn2,*phn3]

    elif '<I_start>' in text:
        assert '<I_end>' in text, text
        text = text.split(">", 1)[1].strip() # remove the audio label
        seg1 = text.split("<I_start>")[0]
        seg2 = text.split("<I_start>")[1].split("<I_end>")[0]
        seg3 = text.split("<I_end>")[1]
        phn1 = text_tokenizer(seg1)
        if len(phn1) > 0:
            phn1.append(" ")
        phn1.append("<I_start>")
        phn1.append(" ")
        phn2 = text_tokenizer(seg2)
        if len(phn2) > 0:
            phn2.append(" ")
        phn2.append("<I_end>")
        phn3 = text_tokenizer(seg3)
        if len(phn3) > 0:
            phn2.append(" ")
        phn = [*phn1,*phn2,*phn3]
            
    else:
        phn = text_tokenizer(text)
        
    phn = [item.replace(' ', '<BLK>') for item in phn]
    phn = [item for item in phn if item in valid_symbols]
    return phn

def estimate_duration_range(text):
    words = text.strip().split()
    num_words = len(words)
    min_duration = num_words / 4.0
    max_duration = num_words / 1.5
    ref_min = num_words / 3.0
    ref_max = num_words / 1.5
    return min_duration, max_duration, ref_min, ref_max

def get_duration(text, predicted_duration):
    cleaned_text = re.sub(r"<[^>]*>", "", text)
    min_dur, max_dur, ref_min, ref_max = estimate_duration_range(cleaned_text)
    event_dur = random.uniform(0.5, 2.0) if "<I_start>" in text else 0
    if predicted_duration < min_dur + event_dur or predicted_duration > max_dur + event_dur:
        return round(random.uniform(ref_min, ref_max), 2) + event_dur
    return predicted_duration

def run(
        model_list, 
        device, 
        duration, 
        transcript, 
        caption,
        speed=1.0,
        steps=25,
        cfg=2.0
    ):
    model, vocoder, phn2num, text_tokenizer, clap_model, duration_tokenizer, duration_model, caption_tokenizer, caption_encoder = model_list
    print("Start Generation...")
    start_time = time.time()
    if "<B_start>" in transcript or "<I_start>" in transcript:
        tag = transcript.split(">", 1)[0].strip()
        tag = tag[1:].lower().replace("_"," ")
    else:
        tag = "none"

    phn = encode(transcript, text_tokenizer)
    text_tokens = [phn2num[item] for item in phn]
    text = torch.LongTensor(text_tokens).unsqueeze(0).to(device)
    if duration is None:
        duration_inputs = caption + " <NEW_SEP> " + transcript
        duration_inputs = duration_tokenizer(duration_inputs, return_tensors="pt", padding="max_length", truncation=True, max_length=400)
    
    with torch.no_grad():
        batch_encoding = caption_tokenizer(caption, return_tensors="pt")
        ori_tokens = batch_encoding["input_ids"].to(device)
        prompt = caption_encoder(input_ids=ori_tokens).last_hidden_state.squeeze().unsqueeze(0).to(device)
        tag_data = [tag] 
        tag_embed = clap_model.get_text_embedding(tag_data, use_tensor=True)
        clap = tag_embed.squeeze().unsqueeze(0).to(device)

        if duration is None:
            duration_outputs = duration_model(**duration_inputs)
            predicted_duration = duration_outputs.logits.squeeze().item()
            duration = get_duration(transcript, predicted_duration)
    if speed == 0:
        speed = 1
    duration = duration / speed
    audio_clips = torch.zeros([1, math.ceil(duration*24000/256), 100]).to(device)
    cond = None
    seq_len_prompt = prompt.shape[1]
    prompt_lens = torch.Tensor([prompt.shape[1]])
    prompt_mask = make_pad_mask(prompt_lens, seq_len_prompt).to(prompt.device)
    gen = sample(model, vocoder,
                 audio_clips, cond, text, prompt, clap, prompt_mask,
                 steps=steps, cfg=cfg,
                 sway_sampling_coef=-1.0, device=device)

    end_time = time.time()
    audio_len = gen.shape[-1] / 24000 # sampling rate fixed in this work
    rtf = (end_time-start_time)/audio_len
    print(f"RTF: {rtf:.4f}")
    return gen

def load_model(device, task):
    print("Downloading model from Huggingface...")
    local_dir = snapshot_download(
        repo_id="OpenSound/CapSpeech-models"
    )
    if task == "PT":
        model_path = os.path.join(local_dir, "nar_PT.pt")
    elif task == "CapTTS":
        model_path = os.path.join(local_dir, "nar_CapTTS.pt")
    elif task == "EmoCapTTS":
        model_path = os.path.join(local_dir, "nar_EmoCapTTS.pt")
    elif task == "AccCapTTS":
        model_path = os.path.join(local_dir, "nar_AccCapTTS.pt")
    elif task == "AgentTTS":
        model_path = os.path.join(local_dir, "nar_AgentTTS.pt")
    else:
        assert 1 == 0, task

    print("Loading models...")
    params = load_yaml_with_includes(os.path.join(local_dir, "nar_pretrain.yaml"))
    model = CrossDiT(**params['model']).to(device)
    checkpoint = torch.load(model_path)['model']
    model.load_state_dict(checkpoint, strict=True)

    # mel spectrogram
    mel = MelSpec(**params['mel']).to(device)
    latent_sr = params['mel']['target_sample_rate'] / params['mel']['hop_length']

    # load vocab
    vocab_fn = os.path.join(os.path.join(local_dir, "vocab.txt"))
    with open(vocab_fn, "r") as f:
        temp = [l.strip().split(" ") for l in f.readlines() if len(l) != 0]
        phn2num = {item[1]:int(item[0]) for item in temp}

    # load g2p
    text_tokenizer = G2p()

    # load vocoder
    # instantiate the model. You can optionally set use_cuda_kernel=True for faster inference.
    vocoder = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False)

    # remove weight norm in the model and set to eval mode
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to(device)

    # load t5
    caption_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    caption_encoder = T5EncoderModel.from_pretrained("google/flan-t5-large").to(device).eval()

    # load clap
    clap_model = laion_clap.CLAP_Module(enable_fusion=False)
    clap_model.load_ckpt(os.path.join(local_dir, "clap-630k-best.pt"))

    # load duration predictor
    duration_tokenizer = AutoTokenizer.from_pretrained(os.path.join(local_dir, "nar_duration_predictor"))
    duration_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(local_dir, "nar_duration_predictor"))
    duration_model.eval()
    model_list = [model, vocoder, phn2num, text_tokenizer, clap_model, duration_tokenizer, duration_model, caption_tokenizer, caption_encoder]

    return model_list
