import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union
import os
import datasets
import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset, IterableDataset, concatenate_datasets, interleave_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoTokenizer
import torchaudio
import torchaudio.transforms as T

@dataclass
class DataCollatorEncodecWithPadding:
    """
    Data collator that will dynamically pad the inputs received to the longest sequence in the batch or
    to `max_length` if `max_length` is set and `padding=max_length`.
    """

    feature_extractor: AutoFeatureExtractor
    audio_column_name: str
    librittsr_dir: Optional[str] = None
    other_dir: Optional[str] = None
    feature_extractor_input_name: Optional[str] = "input_values"
    max_length: Optional[int] = None
    padding: Optional[str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        sampling_rate = self.feature_extractor.sampling_rate
        # load audio
        audios = []
        for f in features:
            path = f[self.audio_column_name]
            source = f["source"]
            if source == "libritts-r":
                path = os.path.join(self.librittsr_dir, path)
            else:
                path = os.path.join(self.other_dir, path)

            if os.path.exists(path):
                waveform, sr = torchaudio.load(path)
                if sr != sampling_rate:
                    resampler = T.Resample(orig_freq=sr, new_freq=sampling_rate)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                audios.append(waveform.squeeze())
            else:
                print(f"Read error: {path}")


        len_audio = [len(audio) for audio in audios]
        if self.max_length is not None:
            audios = [audio[: min(l, self.max_length)] for audio, l in zip(audios, len_audio)]

        # since resampling has already been performed in the 'load_multiple_datasets' function,
        # a fixed sampling_rate(44100hz) is passed to the feature_extractor.
        batch = self.feature_extractor(
            [np.asarray(a, dtype=np.float32) for a in audios], sampling_rate=sampling_rate, return_tensors="pt", padding=self.padding, max_length=self.max_length
        )
        batch["len_audio"] = torch.tensor(len_audio).unsqueeze(1)
        return batch


@dataclass
class DataCollatorParlerTTSWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        prompt_tokenizer (:class:`~transformers.AutoTokenizer`)
            The prompt_tokenizer used for proccessing the data.
        description_tokenizer (:class:`~transformers.AutoTokenizer`)
            The description_tokenizer used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    prompt_tokenizer: AutoTokenizer
    description_tokenizer: AutoTokenizer
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    prompt_max_length: Optional[int] = None
    description_max_length: Optional[int] = None
    audio_max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods

        labels = [torch.tensor(feature["labels"]).transpose(0, 1) for feature in features]
        # (bsz, seq_len, num_codebooks)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        if self.audio_max_length is not None and self.padding == "max_length":
            labels = torch.nn.functional.pad(
                labels, pad=(0, 0, 0, max(self.audio_max_length - labels.shape[1], 0)), value=-100
            )

        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]

        input_ids = self.description_tokenizer.pad(
            input_ids,
            return_tensors="pt",
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.description_max_length,
        )

        batch = {"labels": labels, **input_ids}

        prompt_input_ids = [{"input_ids": feature["prompt_input_ids"]} for feature in features]
        prompt_input_ids = self.prompt_tokenizer.pad(
            prompt_input_ids,
            return_tensors="pt",
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.prompt_max_length,
        )

        batch["prompt_input_ids"] = prompt_input_ids["input_ids"]
        if "attention_mask" in prompt_input_ids:
            batch["prompt_attention_mask"] = prompt_input_ids["attention_mask"]

        return batch


def convert_dataset_str_to_list(
    dataset_names,
    splits=None,
    dataset_samples=None,
    default_split="train",
):
    if isinstance(dataset_names, str):
        dataset_names = dataset_names.split("+")
        splits = splits.split("+") if splits is not None else None
        dataset_samples = dataset_samples.split("+") if dataset_samples is not None else None

    if splits is not None and len(splits) != len(dataset_names):
        raise ValueError(
            f"Ensure one split is passed for each dataset, got {len(dataset_names)} datasets and {len(splits)} splits."
        )

    if dataset_samples is not None:
        if len(dataset_samples) != len(dataset_names):
            raise ValueError(
                f"Ensure one sample is passed for each dataset, got {len(dataset_names)} datasets and "
                f"{len(dataset_samples)} samples."
            )
        dataset_samples = [float(ds_sample) for ds_sample in dataset_samples]
    else:
        dataset_samples = [None] * len(dataset_names)

    splits = splits if splits is not None else [default_split for _ in range(len(dataset_names))]

    dataset_names_dict = []
    for i, ds_name in enumerate(dataset_names):
        dataset_names_dict.append(
            {
                "name": ds_name,
                "split": splits[i],
                "samples": dataset_samples[i],
            }
        )
    return dataset_names_dict


def load_multiple_datasets(
    accelerator: Accelerator,
    dataset_names: Union[List, str],
    splits: Optional[Union[List, str]] = None,
    label_column_names: Optional[List] = None,
    stopping_strategy: Optional[str] = "first_exhausted",
    dataset_samples: Optional[Union[List, np.array]] = None,
    streaming: Optional[bool] = False,
    seed: Optional[int] = None,
    id_column_name: Optional[str] = None,
    columns_to_keep: Optional[Set[str]] = None,
    prompt_column_name: Optional[str] = None,
    sampling_rate: Optional[int] = None,
    audio_column_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    librittsr_dir: Optional[Union[List, str]] = None,
    other_dir: Optional[Union[List, str]] = None,
    **kwargs,
) -> Union[Dataset, IterableDataset]:
    dataset_names_dict = convert_dataset_str_to_list(
        dataset_names, splits, label_column_names, dataset_samples
    )

    if dataset_samples is not None:
        dataset_samples = [ds_dict["samples"] for ds_dict in dataset_names_dict]
        probabilities = np.array(dataset_samples) / np.sum(dataset_samples)
    else:
        probabilities = None

    all_datasets = []
    # iterate over the datasets we want to interleave
    for dataset_dict in tqdm(dataset_names_dict, desc="Combining datasets..."):
        with accelerator.local_main_process_first():
            dataset = load_dataset(
                dataset_dict["name"],
                split=dataset_dict["split"],
                streaming=streaming,
                **kwargs,
            )
            dataset_features = dataset.features.keys()

            if columns_to_keep is not None:
                dataset = dataset.remove_columns(set(dataset_features - columns_to_keep))

            def resolve_path(example):
                path = example["audio_path"]
                source = example["source"]

                if source == "libritts-r":
                    full_path = os.path.join(librittsr_dir, path)
                else:
                    full_path = os.path.join(other_dir, path)

                return os.path.exists(full_path)
                
            dataset = dataset.filter(resolve_path, num_proc=16)

        all_datasets.append(dataset)

    if len(all_datasets) == 1:
        # we have a single dataset so just return it as is
        return all_datasets[0]

    if streaming:
        interleaved_dataset = interleave_datasets(
            all_datasets,
            stopping_strategy=stopping_strategy,
            probabilities=probabilities,
            seed=seed,
        )
    else:
        with accelerator.local_main_process_first():
            interleaved_dataset = concatenate_datasets(all_datasets)

    return interleaved_dataset
