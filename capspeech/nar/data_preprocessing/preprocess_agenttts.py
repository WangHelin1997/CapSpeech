import json
import argparse
from pathlib import Path
from typing import List, Dict, Set
from tqdm import tqdm
import soundfile as sf
from datasets import load_dataset
import logging
import os

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the CapSpeech dataset")
    parser.add_argument('--hub', type=str, required=True, help='Huggingface repo')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the JSON files')
    parser.add_argument('--cache_dir', type=str, required=True, help='Cache directory for datasets')
    parser.add_argument('--wav_dir', type=str, required=True, help='Directories containing WAV files')
    parser.add_argument('--audio_min_length', type=float, default=2.0, help='Minimum audio duration in seconds')
    parser.add_argument('--audio_max_length', type=float, default=20.0, help='Maximum audio duration in seconds')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['train', 'val'],
                        help='List of splits to process')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with limited data processing')
    return parser.parse_args()

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def process_dataset_split(split, dataset_split, args) -> List[Dict]:
    """
    Process a single dataset split and extract relevant records.

    Args:
        split: The name of the split (e.g., 'train').
        dataset_split: The dataset split object.
        args: Parsed command-line arguments.

    Returns:
        A list of dictionaries containing the processed records.
    """
    logging.info(f"Processing split: {split}")
    filelist: List[Dict] = []
    total_duration: float = 0.0
    num_samples: int = len(dataset_split) if not args.debug else 500
    source_path = {
        'capspeech-agentdb': args.wav_dir
    }

    for idx in tqdm(range(num_samples), desc=f"Processing {split}"):
        try:
            data = dataset_split[idx]
        except IndexError:
            logging.warning(f"Index {idx} out of range for split '{split}'. Skipping.")
            continue

        audio_path: str = data.get("audio_path", "")
        duration: float = data.get("speech_duration", 0.0)
        source: str = data.get("source", "")
        audio_path = os.path.join(source_path[source], audio_path)

        if not audio_path:
            logging.warning(f"Missing audio_path at index {idx} in split '{split}'. Skipping.")
            continue

        if not os.path.exists(audio_path):
            logging.warning(f"WAV file does not exist: {audio_path}")
            continue

        if not (args.audio_min_length <= duration <= args.audio_max_length):
            continue

        record: Dict = {
            "segment_id": audio_path.split('/')[-2].replace(" ", "")+"_"+audio_path.split('/')[-1].split('.')[0],
            "audio_path": audio_path,
            "text": data.get('text', ''),
            "caption": data.get('caption', ''),
            "duration": duration,
            "source": source
        }

        filelist.append(record)
        total_duration += duration

    logging.info(f"Total duration for split '{split}': {total_duration / 3600:.2f} hrs.")
    logging.info(f"Total records for split '{split}': {len(filelist)}")
    return filelist


def save_json(filelist: List[Dict], output_path: Path) -> None:
    """
    Save the list of records to a JSON file.

    Args:
        filelist: List of dictionaries containing the records.
        output_path: Path to the output JSON file.
    """
    try:
        with output_path.open('w', encoding='utf-8') as json_file:
            json.dump(filelist, json_file, ensure_ascii=False, indent=4)
        logging.info(f"Saved {len(filelist)} records to '{output_path}'")
    except Exception as e:
        logging.error(f"Failed to save JSON to '{output_path}': {e}")


def main() -> None:
    args = parse_args()
    setup_logging()

    save_dir: Path = Path(args.save_dir)
    jsons_dir: Path = save_dir / 'jsons'
    jsons_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"JSON files will be saved to '{jsons_dir}'")
    logging.info("Loading dataset...")
    try:
        ds = load_dataset(args.hub)
        # ds = load_dataset(args.hub, cache_dir=args.cache_dir)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    splits_to_process = args.splits
    available_splits = set(ds.keys())
    selected_splits = [split for split in splits_to_process if split in available_splits]

    missing_splits = set(splits_to_process) - available_splits
    if missing_splits:
        logging.warning(f"The following splits were not found in the dataset and will be skipped: {missing_splits}")

    for split in selected_splits:
        dataset_split = ds[split]
        filelist = process_dataset_split(split, dataset_split, args)
        output_file: Path = jsons_dir / f"{split}.json"
        save_json(filelist, output_file)


if __name__ == "__main__":
    main()
