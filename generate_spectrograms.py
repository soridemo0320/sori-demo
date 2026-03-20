"""
Generate CQT spectrogram PNGs for each demo sample.

CQT parameters are identical to plot_cqt_midi_overlay.py.
One PNG per sample (left channel = input audio), no MIDI overlay.

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CQT 파라미터 (plot_cqt_midi_overlay.py와 동일) ──────────────────
CQT_FMIN         = librosa.midi_to_hz(21)   # A0 = 27.5 Hz
CQT_BINS_PER_OCT = 36                        # 반음 1개 = 3 bins
CQT_N_BINS       = 88 * 3                    # A0(21) ~ C8(108)
CQT_SR           = 16000
CQT_HOP_LENGTH   = 160                       # 10ms @ 16kHz
T_MAX            = 5.0                       # 최대 표시 시간 (초)

# y축 tick (plot_cqt_midi_overlay.py와 동일)
MIDI_TICKS   = [21, 36, 48, 60, 72, 84, 96, 108]
BIN_TICKS    = [(p - 21) * 3 for p in MIDI_TICKS]
HZ_LABELS    = [f"{librosa.midi_to_hz(p):.0f} Hz" for p in MIDI_TICKS]
PITCH_LABELS = [f"{librosa.midi_to_note(p)} ({p})" for p in MIDI_TICKS]


def generate_spectrogram(wav_path: str, output_path: str):
    """
    stereo WAV에서 left channel(=input audio)을 추출해 CQT 스펙트로그램 PNG 저장.
    """
    y, _ = librosa.load(wav_path, sr=CQT_SR, mono=False)
    y_left = y[0] if y.ndim == 2 else y   # left channel

    # CQT 계산
    C = librosa.cqt(
        y_left,
        sr=CQT_SR,
        hop_length=CQT_HOP_LENGTH,
        fmin=CQT_FMIN,
        n_bins=CQT_N_BINS,
        bins_per_octave=CQT_BINS_PER_OCT,
    )
    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)  # (n_bins, T)

    duration = y_left.shape[0] / CQT_SR
    t_max = min(duration, T_MAX)

    # 시각화
    fig, ax1 = plt.subplots(figsize=(7, 4), dpi=150)
    ax1.imshow(
        C_db,
        aspect="auto",
        origin="lower",
        interpolation="none",
        cmap="magma",
        extent=[0, t_max, 0, CQT_N_BINS - 1],
        vmin=C_db.max() - 80,
        vmax=C_db.max(),
    )
    ax1.set_ylabel("Frequency (Hz)", fontsize=10)
    ax1.set_yticks(BIN_TICKS)
    ax1.set_yticklabels(HZ_LABELS, fontsize=8)
    ax1.set_ylim(0, CQT_N_BINS - 1)
    ax1.set_xlabel("Time (s)", fontsize=10)
    ax1.set_xlim(0, t_max)

    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(BIN_TICKS)
    ax2.set_yticklabels(PITCH_LABELS, fontsize=8)
    ax2.set_ylabel("MIDI Pitch", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate CQT spectrogram PNGs for demo page.")
    parser.add_argument(
        "--audio_dir", default=".",
        help="Root dir containing sample subdirs (e.g. 000852_4.0/). Default: current dir"
    )
    parser.add_argument(
        "--output_dir", default="images",
        help="Output directory for PNG files. Default: ./images"
    )
    args = parser.parse_args()

    # sample 서브디렉토리 탐색 (숫자로 시작하는 폴더)
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
        sample_id = sample_dir.split("_")[0]  # e.g., "000852"
        dir_path = os.path.join(args.audio_dir, sample_dir)

        # 첫 번째 WAV 파일 사용 (left channel = input audio로 모두 동일)
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
