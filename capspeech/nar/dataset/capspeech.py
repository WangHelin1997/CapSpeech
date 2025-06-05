# @ hwang258@jhu.edu
import os
import json
import torch
import random
import logging
import shutil
import typing as tp
import numpy as np
import torchaudio
import sys
from torch.utils.data import Dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


class CapSpeech(Dataset):
    def __init__(
        self,
        dataset_dir: str = None,
        clap_emb_dir: str = None,
        t5_folder_name: str = "t5",
        phn_folder_name: str = "g2p",
        manifest_name: str = "manifest",
        json_name: str = "jsons",
        dynamic_batching: bool = True,
        text_pad_token: int = -1,
        audio_pad_token: float = 0.0,
        split: str = "val",
        sr: int = 24000,
        norm_audio: bool = False,
        vocab_file: str = None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.clap_emb_dir = clap_emb_dir
        self.t5_folder_name = t5_folder_name
        self.phn_folder_name = phn_folder_name
        self.manifest_name = manifest_name
        self.json_name = json_name
        self.dynamic_batching = dynamic_batching
        self.text_pad_token = text_pad_token
        self.audio_pad_token = torch.tensor(audio_pad_token)
        self.split = split
        self.sr = sr
        self.norm_audio = norm_audio

        assert self.split in ['train', 'train_small', 'val', 'test']
        manifest_fn = os.path.join(self.dataset_dir, self.manifest_name, self.split+".txt")

        meta = read_json(os.path.join(self.dataset_dir, self.json_name, self.split + ".json"))
        self.meta = {item["segment_id"]: item["audio_path"] for item in meta}

        with open(manifest_fn, "r") as rf:
            data = [l.strip().split("\t") for l in rf.readlines()]

        # data = [item for item in data if item[2] == 'none'] # remove sound effects

        self.data = [item[0] for item in data]
        self.tag_list = [item[1] for item in data]

        logging.info(f"number of data points for {self.split} split: {len(self.data)}")

        # phoneme vocabulary
        if vocab_file is None:
            vocab_fn = os.path.join(self.dataset_dir, "vocab.txt")
        else:
            vocab_fn = vocab_file
        with open(vocab_fn, "r") as f:
            temp = [l.strip().split(" ") for l in f.readlines() if len(l) != 0]
            self.phn2num = {item[1]:int(item[0]) for item in temp}

    def __len__(self):
        return len(self.data)

    def _load_audio(self, audio_path):
        try:
            y, sr = torchaudio.load(audio_path)
            if y.shape[0] > 1:
                y = y.mean(dim=0, keepdim=True)
            if sr != self.sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)
                y = resampler(y)
            if self.norm_audio:
                eps = 1e-9
                max_val = torch.max(torch.abs(y))
                y = y / (max_val + eps)
            if torch.isnan(y.mean()):
                return None
            return y
        except:
            return None

    def _load_phn_enc(self, index):
        try:
            seg_id = self.data[index]
            pf = os.path.join(self.dataset_dir, self.phn_folder_name, seg_id+".txt")
            audio_path = self.meta[seg_id]
            cf = os.path.join(self.dataset_dir, self.t5_folder_name, seg_id+".npz")
            tagf = os.path.join(self.clap_emb_dir, self.tag_list[index]+'.npz')
            with open(pf, "r") as p:
                phns = [l.strip() for l in p.readlines()]
                assert len(phns) == 1, phns
                x = [self.phn2num[item] for item in phns[0].split(" ")]
            c = np.load(cf)['arr_0']
            c = torch.tensor(c).squeeze()
            tag = np.load(tagf)['arr_0']
            tag = torch.tensor(tag).squeeze()
            y = self._load_audio(audio_path)
            if y is not None:
                return x, y, c, tag
            return None, None, None, None
        except:
            return None, None, None, None

    def __getitem__(self, index):
        x, y, c, tag = self._load_phn_enc(index)
        if x is None:
            return {
                "x": None,
                "x_len": None,
                "y": None,
                "y_len": None,
                "c": None,
                "c_len": None,
                "tag": None
            }
        x_len, y_len, c_len = len(x), len(y[0]), len(c)
        y_len = y_len / self.sr

        if y_len * self.sr / 256 <= x_len:
            return {
                "x": None,
                "x_len": None,
                "y": None,
                "y_len": None,
                "c": None,
                "c_len": None,
                "tag": None
            }
            
        x = torch.LongTensor(x)
        return {
            "x": x,
            "x_len": x_len,
            "y": y,
            "y_len": y_len,
            "c": c,
            "c_len": c_len,
            "tag": tag
        }

    def collate(self, batch):
        out = {key:[] for key in batch[0]}
        for item in batch:
            if item['x'] == None: # deal with load failure
                continue
            if item['c'].ndim != 2:
                continue
            for key, val in item.items():
                out[key].append(val)

        res = {}
        res["x"] = torch.nn.utils.rnn.pad_sequence(out["x"], batch_first=True, padding_value=self.text_pad_token)
        res["x_lens"] = torch.LongTensor(out["x_len"])
        if self.dynamic_batching:
            res['y'] = torch.nn.utils.rnn.pad_sequence([item.transpose(1,0) for item in out['y']],padding_value=self.audio_pad_token)
            res['y'] = res['y'].permute(1,2,0) # T B K -> B K T
        else:
            res['y'] = torch.stack(out['y'], dim=0)

        res["y_lens"] = torch.Tensor(out["y_len"])
        res['c'] = torch.nn.utils.rnn.pad_sequence(out['c'], batch_first=True)
        res["c_lens"] = torch.LongTensor(out["c_len"])
        res["tag"] = torch.stack(out['tag'], dim=0)
        return res


if __name__ == "__main__":    
    # debug
    import argparse
    from torch.utils.data import DataLoader
    from accelerate import Accelerator

    dataset = CapSpeech(
        dataset_dir="./data/capspeech",
        clap_emb_dir="./data/clap_embs/",
        split="val"
    )
