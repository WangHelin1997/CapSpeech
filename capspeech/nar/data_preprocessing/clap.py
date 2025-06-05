import torch
import laion_clap
import os
from tqdm import tqdm
import numpy as np

with open("events.txt", "r") as f:
    events = [line.strip() for line in f]
    
save_path = './clap_embs'

model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt("./630k-best.pt")

with torch.no_grad():
    for event in tqdm(events):
        text_data = [event.lower()] 
        text_embed = model.get_text_embedding(text_data, use_tensor=True)
        text_embed = text_embed.squeeze().cpu().numpy()
        save_fn = os.path.join(save_path, event.lower().replace(" ", "_")+".npz")
        np.savez_compressed(save_fn, text_embed)



