"""Run ETTm1 experiments for multiple online methods and horizons."""

import datetime
import subprocess
import sys


BASE_ARGS = [
    "--dataset", "ETTm1",
    "--border_type", "online",
    "--model", "iTransformer",
    "--seq_len", "288",
    "--itr", "1",
    "--only_test",
    "--pretrain",
    "--wo_valid",
]

PRED_LENS = [24, 48, 96]
DEFAULT_LR = "1e-5"


def build_command(method, pred_len):
    cmd = [sys.executable, "-u", "run.py"]
    cmd += BASE_ARGS + ["--pred_len", str(pred_len)]

    if method != "Frozen":
        cmd += ["--online_method", method]
        cmd += ["--online_learning_rate", DEFAULT_LR]

    if method == "HMem":
        cmd += [
            "--use_snma", "False",
            "--use_chrc", "True",
            "--retrieval_top_k", "5",
            "--chrc_aggregation", "softmax",
            "--chrc_trust_threshold", "0.5",
            "--chrc_gate_steepness", "10.0",
            "--chrc_use_horizon_mask", "True",
            "--chrc_horizon_mask_mode", "exp",
            "--chrc_horizon_mask_decay", "0.98",
            "--chrc_horizon_mask_min", "0.2",
            "--chrc_use_buckets", "True",
            "--chrc_bucket_num", "4",
        ]

    return cmd


def run_and_log(cmd, log_fp):
    log_fp.write("\n$ " + " ".join(cmd) + "\n")
    log_fp.flush()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        log_fp.write(line)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def main():
    methods = [
        "Frozen",
        "Online",
        "ER",
        "DERpp",
        "ACL",
        "CLSER",
        "MIR",
        "SOLID",
        "HMem",
    ]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"ettm1_exp_{timestamp}.log"
    print("Logging to:", log_path)
    with open(log_path, "w", encoding="utf-8") as log_fp:
        for pred_len in PRED_LENS:
            for method in methods:
                cmd = build_command(method, pred_len)
                print("\nRunning:", " ".join(cmd))
                run_and_log(cmd, log_fp)


if __name__ == "__main__":
    main()
