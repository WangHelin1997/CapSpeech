# @ hwang258@jh.edu

import argparse
import logging
import json
import glob
import os
import numpy as np
import tqdm
import time
import multiprocessing
from g2p_en import G2p
import nltk
nltk.download('averaged_perceptron_tagger_eng')

def parse_args():
    parser = argparse.ArgumentParser(description="Encode the gigaspeech phonemes using g2p model")
    parser.add_argument('--save_dir', type=str, default=None, help="path to the manifest, phonemes, and encodec codes dirs")
    parser.add_argument('--num_cpus', type=int, default=10)
    return parser.parse_args()

if __name__ == "__main__":
    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = parse_args()

    # get the path
    phn_save_root = os.path.join(args.save_dir, "g2p")
    os.makedirs(phn_save_root, exist_ok=True)

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

    ### phonemization
    text_tokenizer = G2p()

    stime = time.time()

    logging.info(f"phonemizing...")
    json_paths = glob.glob(os.path.join(args.save_dir, 'jsons', '*.json'))
    for json_path in json_paths:
        with open(json_path, 'r') as json_file:
            jsondata = json.load(json_file)

        df_split = np.array_split(jsondata, args.num_cpus)
        print(len(jsondata))
        # Optional: Save each part to a separate JSON file
        cmds = []
        for idx, part in enumerate(df_split):
            cmds.append((idx, part))

        def process_one(indx, splitdata):
            for key in tqdm.tqdm(range(len(splitdata))):
                save_fn = os.path.join(phn_save_root, splitdata[key]['segment_id']+".txt")
                if not os.path.exists(save_fn):
                    text = splitdata[key]['text']
                    
                    if splitdata[key]['source'] == "libritts-r":
                        
                        text = text.split(">", 1)[1].strip() # remove the audio label
                        if "<B_start>" in text:
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
                            
                        elif "<I_start>" in text:
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
                    wrong_phn = [item for item in phn if item not in valid_symbols]
                    if len(wrong_phn) > 0:
                        print(wrong_phn)
                    phn_seq = " ".join(phn)
                    with open(save_fn, "w") as f:
                        f.write(phn_seq)
                
        with multiprocessing.Pool(processes=args.num_cpus) as pool:
            pool.starmap(process_one, cmds)