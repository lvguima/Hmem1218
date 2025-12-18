#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to automatically run H-Mem diagnostic experiments
Output is saved to log files
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path


class TeeOutput:
    """Writes output to both file and terminal"""
    def __init__(self, file_path):
        self.file = open(file_path, 'a', encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
    
    def write(self, text):
        self.file.write(text)
        self.file.flush()
        self.stdout.write(text)
        self.stdout.flush()
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        self.file.close()


def run_experiment(exp_num, exp_name, command, log_file):
    """Runs a single experiment"""
    print(f"\n{'='*80}")
    print(f"Start Experiment {exp_num}: {exp_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Command: {command}")
    print(f"{ '='*80}\n")
    
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"Start Experiment {exp_num}: {exp_name}\n")
    log_file.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Command: {command}\n")
    log_file.write(f"{ '='*80}\n\n")
    log_file.flush()
    
    try:
        # Execute command
        # Use unbuffered binary mode to capture output exactly as is (preserving \r)
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0  # Unbuffered
        )
        
        # Read byte by byte to preserve cursor movements for tqdm
        while True:
            char = process.stdout.read(1)
            if not char and process.poll() is not None:
                break
            
            if char:
                # Write directly to terminal buffer to handle \r correctly
                sys.stdout.buffer.write(char)
                sys.stdout.flush()
                
                # Write to log file (decode best effort)
                try:
                    text = char.decode('utf-8', errors='ignore')
                    log_file.write(text)
                    # Don't flush log file on every char for performance
                    if char == b'\n':
                        log_file.flush()
                except:
                    pass
        
        return_code = process.wait()
        
        if return_code == 0:
            print(f"\n[OK] Experiment {exp_num} Completed Successfully")
            log_file.write(f"\n[OK] Experiment {exp_num} Completed Successfully\n")
        else:
            print(f"\n[FAIL] Experiment {exp_num} Failed, Return Code: {return_code}")
            log_file.write(f"\n[FAIL] Experiment {exp_num} Failed, Return Code: {return_code}\n")
        
        log_file.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.flush()
        
        return return_code == 0
        
    except Exception as e:
        error_msg = f"\n[FAIL] Experiment {exp_num} Execution Error: {str(e)}\n"
        print(error_msg)
        log_file.write(error_msg)
        log_file.flush()
        return False


def main():
    # Define all experiments
    experiments = [
        {
            "num": 1,
            "name": "Pure SNMA (No CHRC) - Check SNMA baseline",
            "command": "python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 0.0001 --lora_rank 8 --lora_alpha 16.0 --memory_dim 256 --bottleneck_dim 32 --hmem_warmup_steps 100 --freeze True --use_snma True --use_chrc False"
        },
        {
            "num": 2,
            "name": "Pure CHRC (No SNMA) - Retrieval Only, High Capacity, Low LR",
            "command": "python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 0.00001 --memory_capacity 2000 --retrieval_top_k 5 --hmem_warmup_steps 100 --freeze True --use_snma False --use_chrc True"
        },
        {
            "num": 3,
            "name": "H-Mem Full (High Capacity + Low LR) - Recommended Fix",
            "command": "python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 0.00001 --lora_rank 16 --lora_alpha 32.0 --memory_dim 256 --bottleneck_dim 32 --memory_capacity 2000 --retrieval_top_k 5 --hmem_warmup_steps 100 --freeze True --use_snma True --use_chrc True"
        },
        {
            "num": 4,
            "name": "H-Mem Full (Aggressive Retrieval - Larger Top-K & Capacity)",
            "command": "python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 0.00001 --lora_rank 8 --memory_capacity 4000 --retrieval_top_k 10 --hmem_warmup_steps 100 --freeze True --use_snma True --use_chrc True"
        },
        {
            "num": 5,
            "name": "ER Baseline (Reproduce for Fairness)",
            "command": "python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method ER --only_test --pretrain --online_learning_rate 3e-7 --pin_gpu True --save_opt"
        },
        {
            "num": 6,
            "name": "H-Mem Full (High Rank LoRA)",
            "command": "python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 0.00005 --lora_rank 32 --lora_alpha 64.0 --memory_dim 512 --bottleneck_dim 64 --memory_capacity 2000 --hmem_warmup_steps 100 --freeze True --use_snma True --use_chrc True"
        },
        {
            "num": 7,
            "name": "H-Mem Full (CHRC weighted_mean aggregation)",
            "command": "python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 0.00001 --lora_rank 8 --lora_alpha 16.0 --memory_dim 256 --bottleneck_dim 32 --memory_capacity 2000 --retrieval_top_k 5 --hmem_warmup_steps 100 --freeze True --use_snma True --use_chrc True --chrc_aggregation weighted_mean"
        },
        {
            "num": 8,
            "name": "H-Mem Full (CHRC softmax aggregation with temp=0.5)",
            "command": "python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 0.00001 --lora_rank 8 --lora_alpha 16.0 --memory_dim 256 --bottleneck_dim 32 --memory_capacity 2000 --retrieval_top_k 5 --hmem_warmup_steps 100 --freeze True --use_snma True --use_chrc True --chrc_temperature 0.5"
        },
    ]
    
    # Create log directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = log_dir / f"hmem_diagnostic_experiments_{timestamp}.log"
    
    print(f"\n{'#'*80}")
    print(f"Starting all {len(experiments)} H-Mem diagnostic experiments")
    print(f"Log file: {log_file_path}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{ '#'*80}\n")
    
    # Open log file
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"{ '#'*80}\n")
        log_file.write(f"H-Mem Diagnostic Experiments Log\n")
        log_file.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Total Experiments: {len(experiments)}\n")
        log_file.write(f"{ '#'*80}\n\n")
        log_file.flush()
        
        # Statistics
        results = []
        
        # Run each experiment sequentially
        for exp in experiments:
            success = run_experiment(
                exp["num"],
                exp["name"],
                exp["command"],
                log_file
            )
            results.append({
                "num": exp["num"],
                "name": exp["name"],
                "success": success
            })
        
        # Print summary
        print(f"\n{'#'*80}")
        print("Experiment Run Summary")
        print(f"{ '#'*80}\n")
        
        log_file.write(f"\n{'#'*80}\n")
        log_file.write("Experiment Run Summary\n")
        log_file.write(f"{ '#'*80}\n\n")
        
        success_count = sum(1 for r in results if r["success"])
        fail_count = len(results) - success_count
        
        for result in results:
            status = "[OK] Success" if result["success"] else "[FAIL] Failed"
            summary = f"Exp {result['num']:2d}: {status} - {result['name']}"
            print(summary)
            log_file.write(summary + "\n")
        
        print(f"\nTotal: {len(results)} Experiments, Success: {success_count}, Failed: {fail_count}")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {log_file_path}\n")
        
        log_file.write(f"\nTotal: {len(results)} Experiments, Success: {success_count}, Failed: {fail_count}\n")
        log_file.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.flush()
    
    print(f"All experiments completed! Log saved to: {log_file_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nUser interrupted experiment execution")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nScript execution error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
