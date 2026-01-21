"""
Creates a webdataset for training a HiFiGAN with SSL features. Each sample comprises an
audio file and its corresponding SSL features.

Both folders should have the same folder structure. We iterate over the SSl folder and
expect to find the corresponding audio at the same relative location. We flatten the
directory structure, so make sure each file has a unique name.
"""

import argparse
import os
import shutil

from tqdm import tqdm


def prepare_tar(dump_dir: str, audio_dir: str, ssl_dir: str, ext: str) -> str:
    os.makedirs(dump_dir)
    for folder, _, files in tqdm(os.walk(ssl_dir)):
        for f in files:
            if not f.endswith(".pt"):
                continue

            ssl_path = os.path.join(folder, f)
            shutil.copyfile(ssl_path, os.path.join(dump_dir, f))

            audio_path = ssl_path.replace(ssl_dir, audio_dir).replace(".pt", ext)
            shutil.copyfile(audio_path, os.path.join(dump_dir, f.replace(".pt", ext)))

    tar_file = f"{dump_dir}.tar"
    os.system(
        f"find {dump_dir} -print0 | sort -z | tar cf {tar_file} --no-recursion --null -T -"
    )
    return tar_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a webdataset with SSL features and audios"
    )
    parser.add_argument("dump_dir")
    parser.add_argument("audio_dir")
    parser.add_argument("ssl_dir")
    parser.add_argument("--ext", default=".flac", type=str)
    args = parser.parse_args()
    tar_file = prepare_tar(args.dump_dir, args.audio_dir, args.ssl_dir, args.ext)
    print(f"Tar file created at {tar_file}")
