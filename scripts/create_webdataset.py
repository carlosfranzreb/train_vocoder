import argparse
import os
import webdataset as wds
from tqdm import tqdm


def prepare_shards(
    output_dir: str, audio_dir: str, ssl_dir: str, n_tars: int, ext: str
):
    """
    Creates multiple webdataset shards by pairing SSL features and audio files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # gather all valid pairs
    file_pairs = []
    for folder, _, files in tqdm(os.walk(ssl_dir), desc="Searching for file pairs..."):
        for f in files:
            if not f.endswith(".pt"):
                continue

            ssl_path = os.path.join(folder, f)
            audio_path = ssl_path.replace(ssl_dir, audio_dir).replace(".pt", ext)

            if os.path.exists(audio_path):
                # We store the base name (key) and the full paths
                key = f.replace(".pt", "")
                file_pairs.append((key, ssl_path, audio_path))
            else:
                print(f"Warning: Audio missing for {ssl_path}")

    # create the pattern for filenames (e.g., shard-00001.tar)
    print(f"{len(file_pairs)} samples will be written across {n_tars} shards")
    writers = [
        wds.TarWriter(os.path.join(output_dir, f"shard-{i:05d}.tar"))
        for i in range(n_tars)
    ]

    # distribute samples across writers
    for i, (key, ssl_path, audio_path) in tqdm(
        enumerate(file_pairs), desc="Writing pairs"
    ):
        writer = writers[i % n_tars]

        with open(ssl_path, "rb") as f:
            ssl_data = f.read()
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        # WebDataset uses the keys to group files into a single sample
        sample = {"__key__": key, "pt": ssl_data, ext.lstrip("."): audio_data}
        writer.write(sample)

    # close  writers to finalize the tar files
    for writer in writers:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create sharded WebDataset")
    parser.add_argument("output_dir", help="Directory to save .tar files")
    parser.add_argument("audio_dir")
    parser.add_argument("ssl_dir")
    parser.add_argument("n_tars", type=int, help="Number of shards to create")
    parser.add_argument("--ext", default=".flac", type=str)
    args = parser.parse_args()

    prepare_shards(args.output_dir, args.audio_dir, args.ssl_dir, args.n_tars, args.ext)
    print(f"\nDone! Shards are located in: {args.output_dir}")
