# @ hwang258@jh.edu

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Create the vocab set of gigaspeech")
    parser.add_argument('--save_dir', type=str, default=None, help="path to the manifest, phonemes, and encodec codes dirs")
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    savepath = os.path.join(args.save_dir, 'vocab.txt')
    phn_vocab = []
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


    phn_vocab = set(valid_symbols)
    
    with open(savepath, "w") as f:
        for i, phn in enumerate(list(phn_vocab)):
            if i < len(list(phn_vocab)) - 1:
                f.write(f"{str(i)} {phn}\n")
            else:
                f.write(f"{str(i)} {phn}")