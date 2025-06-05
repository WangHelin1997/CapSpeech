# @ hwang258@jh.edu

import os
import argparse
from tqdm import tqdm
import json
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Create manifests for gigaspeech")
    parser.add_argument('--save_dir', type=str, default=None, help="path to the manifest, phonemes, and encodec codes dirs")

    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    phn_save_root = os.path.join(args.save_dir, "g2p")
    t5_save_root = os.path.join(args.save_dir, "t5")
    manifest_root = os.path.join(args.save_dir, "manifest")
    os.makedirs(manifest_root, exist_ok=True)
    
    json_paths = glob.glob(os.path.join(args.save_dir, 'jsons', '*.json'))
    for json_path in json_paths:
        savelines = []
        with open(json_path, 'r') as json_file:
            jsondata = json.load(json_file)
        for key in tqdm(range(len(jsondata))):
            if os.path.exists(os.path.join(phn_save_root, jsondata[key]['segment_id']+".txt")) and \
                os.path.exists(os.path.join(t5_save_root, jsondata[key]['segment_id']+".npz")):
                    if jsondata[key]['source'] == 'libritts-r':
                        tag = jsondata[key]['text'].split(">", 1)[0].replace("<","").strip()
                    else:
                        tag = "none"
                    savelines.append([jsondata[key]['segment_id'], tag])

        outputlines = ''
        for i in range(len(savelines)):
            outputlines += savelines[i][0]+'\t'+str(savelines[i][1])+'\n'
        with open(os.path.join(manifest_root, json_path.split('/')[-1].replace('.json', '')+'.txt'), "w") as f:
            f.write(outputlines)