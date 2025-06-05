import os
import torch
import librosa
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from torchdiffeq import odeint
from einops import rearrange
from capspeech.nar.utils import make_pad_mask

@torch.no_grad()
def sample(model, vocoder,
           x, cond, text, prompt, clap, prompt_mask,
           steps=25, cfg=2.0,
           sway_sampling_coef=-1.0, device='cuda'):

    model.eval()
    vocoder.eval()

    y0 = torch.randn_like(x)

    neg_text = torch.ones_like(text) * -1
    neg_clap = torch.zeros_like(clap)
    neg_prompt = torch.zeros_like(prompt)
    neg_prompt_mask = torch.zeros_like(prompt_mask)
    neg_prompt_mask[:, 0] = 1

    def fn(t, x):
        pred = model(x=x, cond=cond, text=text, time=t, 
                     prompt=prompt, clap=clap,
                     mask=None,
                     prompt_mask=prompt_mask)

        null_pred = model(x=x, cond=cond, text=neg_text, time=t, 
                          prompt=neg_prompt, clap=neg_clap,
                          mask=None,
                          prompt_mask=neg_prompt_mask)
        return pred + (pred - null_pred) * cfg

    t_start = 0
    t = torch.linspace(t_start, 1, steps, device=device)
    if sway_sampling_coef is not None:
        t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

    trajectory = odeint(fn, y0, t, method="euler")
    out = trajectory[-1]
    out = rearrange(out, 'b n d -> b d n')

    with torch.inference_mode():
        wav_gen = vocoder(out)
    wav_gen_float = wav_gen.squeeze().cpu().numpy() # wav_gen is FloatTensor with shape [1, T_time]
    return wav_gen_float


def prepare_batch(batch, mel, latent_sr):
    x, x_lens, y, y_lens, c, c_lens, tag = batch["x"], batch["x_lens"], batch["y"], batch["y_lens"], batch["c"], batch["c_lens"], batch["tag"]

    # add len for clap embedding
    x_lens = x_lens + 1

    with torch.no_grad():
        audio_clip = mel(y)
        audio_clip = rearrange(audio_clip, 'b d n -> b n d')
        y_lens = (y_lens * latent_sr).long()

    return x, x_lens, audio_clip, y_lens, c, c_lens, tag

# use ground truth duration for simple inference
@torch.no_grad()
def eval_model(model, vocos, mel, val_loader, params,
               steps=25, cfg=2.0,
               sway_sampling_coef=-1.0, device='cuda',
               epoch=0, save_path='logs/eval/', val_num=5):

    save_path = save_path + '/' + str(epoch) + '/'
    os.makedirs(save_path, exist_ok=True)

    latent_sr = params['mel']['target_sample_rate'] / params['mel']['hop_length']

    for step, batch in enumerate(tqdm(val_loader)):
        (text, text_lens, audio_clips, audio_lens, prompt, prompt_lens, clap) = prepare_batch(batch, mel, latent_sr)
        cond = None

        seq_len_prompt = prompt.shape[1]
        prompt_mask = make_pad_mask(prompt_lens, seq_len_prompt).to(prompt.device)

        gen = sample(model, vocos,
                     audio_clips, cond, text, prompt, clap, prompt_mask,
                     steps=steps, cfg=cfg,
                     sway_sampling_coef=sway_sampling_coef, device=device)

        sf.write(save_path + f'{step}.wav', gen, samplerate=params['mel']['target_sample_rate'])

        if step + 1 >= val_num:
            break