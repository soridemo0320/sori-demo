"""
Generate STFT spectrogram PNGs for each demo sample.

One PNG per sample (left channel = input audio).

Usage (run from the demo repo root):
    python generate_spectrograms.py
    python generate_spectrograms.py --audio_dir /path/to/audio --output_dir images
"""

import os
import sys
import glob
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR         = 22050
HOP_LENGTH = 512
N_FFT      = 2048
FMAX       = 4000   # 보컬 범위에 맞게 4kHz까지만 표시
T_MAX      = 5.0


def generate_spectrogram(wav_path: str, output_path: str):
    """
    stereo WAV에서 left channel(=input audio)을 추출해 STFT 스펙트로그램 PNG 저장.
    """
    y, _ = librosa.load(wav_path, sr=SR, mono=False)
    y_left = y[0] if y.ndim == 2 else y

    t_max = min(y_left.shape[0] / SR, T_MAX)
    y_left = y_left[:int(t_max * SR)]

    D = librosa.stft(y_left, n_fft=N_FFT, hop_length=HOP_LENGTH)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # FMAX 이하 빈만 잘라냄
    bin_max = int(FMAX / (SR / 2) * (N_FFT // 2 + 1)) + 1
    D_db = D_db[:bin_max, :]

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    librosa.display.specshow(
        D_db, sr=SR, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="linear",
        fmax=FMAX,
        ax=ax, cmap="magma",
    )
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", fontsize=9)
    ax.set_xticks(np.arange(0.0, T_MAX + 0.01, 0.5))
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate STFT spectrogram PNGs for demo page.")
    parser.add_argument(
        "--audio_dir", default=".",
        help="Root dir containing sample subdirs (e.g. 000852_4.0/). Default: current dir"
    )
    parser.add_argument(
        "--output_dir", default="images",
        help="Output directory for PNG files. Default: ./images"
    )
    args = parser.parse_args()

    sample_dirs = sorted([
        d for d in os.listdir(args.audio_dir)
        if os.path.isdir(os.path.join(args.audio_dir, d)) and d[0].isdigit()
    ])

    if not sample_dirs:
        print(f"ERROR: No sample directories found in '{args.audio_dir}'")
        print("Expected directories like: 000852_4.0, 001239_4.0, ...")
        sys.exit(1)

    print(f"Found {len(sample_dirs)} sample(s): {sample_dirs}\n")

    for sample_dir in sample_dirs:
        sample_id = sample_dir.split("_")[0]
        dir_path = os.path.join(args.audio_dir, sample_dir)

        wav_files = sorted(glob.glob(os.path.join(dir_path, "*.wav")))
        if not wav_files:
            print(f"  WARNING: No WAV files in {dir_path}, skipping")
            continue

        wav_path = wav_files[0]
        output_path = os.path.join(args.output_dir, f"{sample_id}_spec.png")
        print(f"[{sample_dir}]  src: {os.path.basename(wav_path)}")
        generate_spectrogram(wav_path, output_path)

    print(f"\nDone! {len(sample_dirs)} spectrogram(s) saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
