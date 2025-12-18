"""
iTransformer Pretraining Script for ETTm1 (Python Version)
Cross-platform alternative to ETTm1.sh
"""

import os
import sys
import subprocess
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.chdir(PROJECT_ROOT)

print("=" * 60)
print("iTransformer Pretraining on ETTm1")
print("=" * 60)
print()

# Create log directories
log_dir = PROJECT_ROOT / "logs" / "online"
log_dir.mkdir(parents=True, exist_ok=True)

# Settings
seq_len = 512
data = 'ETTm1'
model_name = 'iTransformer'
learning_rate = 0.0001
pred_lens = [24, 48, 96]

print(f"Dataset: {data}")
print(f"Model: {model_name}")
print(f"Sequence Length: {seq_len}")
print(f"Learning Rate: {learning_rate}")
print(f"Prediction Horizons: {pred_lens}")
print(f"Iterations: 3")
print()

for pred_len in pred_lens:
    print("=" * 60)
    print(f"Running pred_len={pred_len}")
    print("=" * 60)
    print()

    log_file = f"logs/online/{model_name}_{data}_{pred_len}_lr{learning_rate}.log"

    cmd = [
        sys.executable, "-u", "run.py",
        "--dataset", data,
        "--border_type", "online",
        "--model", model_name,
        "--seq_len", str(seq_len),
        "--pred_len", str(pred_len),
        "--itr", "3",
        "--only_test",
        "--save_opt",
        "--batch_size", "32",
        "--learning_rate", str(learning_rate),
    ]

    print(f"Command: {' '.join(cmd[:8])}...")
    print(f"Log file: {log_file}")
    print()

    try:
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end='')
                log.write(line)

            process.wait()

            if process.returncode != 0:
                print(f"\n[ERROR] Failed with return code: {process.returncode}")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Stopping experiment...")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

print()
print("=" * 60)
print("All experiments completed successfully!")
print("=" * 60)
