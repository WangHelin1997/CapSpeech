import argparse
import logging
import json
import os
import numpy as np
import torch
import tqdm
import time
from transformers import T5EncoderModel, AutoTokenizer
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Encode the data captionings using t5 model")
    parser.add_argument('--save_dir', type=str, default=None, help="path to the manifest, phonemes, and encodec codes dirs")
    parser.add_argument('--start', type=int, default=0, help='start index for parallel processing')
    parser.add_argument('--end', type=int, default=10000000, help='end index for parallel processing')
    return parser.parse_args()

if __name__ == "__main__":

    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    caption_encoder = T5EncoderModel.from_pretrained("google/flan-t5-large").cuda().eval()

    # get the path
    phn_save_root = os.path.join(args.save_dir, "t5")
    os.makedirs(phn_save_root, exist_ok=True)
    
    stime = time.time()

    logging.info(f"captioning...")
    json_paths = glob.glob(os.path.join(args.save_dir, 'jsons', '*.json'))
    for json_path in json_paths:
        with open(json_path, 'r', encoding="utf-8") as json_file:
            jsondata = json.load(json_file)
            
        jsondata = jsondata[args.start:args.end]

        for key in tqdm.tqdm(range(len(jsondata))):
            save_fn = os.path.join(phn_save_root, jsondata[key]['segment_id']+".npz")
            if not os.path.exists(save_fn):
                text = jsondata[key]['caption']
                
                with torch.no_grad():
                    batch_encoding = tokenizer(text, return_tensors="pt")
                    ori_tokens = batch_encoding["input_ids"].cuda()
                    outputs = caption_encoder(input_ids=ori_tokens).last_hidden_state
                
                phn = outputs.cpu().numpy()
                np.savez_compressed(save_fn, phn)