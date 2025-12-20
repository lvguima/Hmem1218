
1. python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma False --use_chrc True --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.3 --chrc_gate_steepness 10.0
[H-Mem] Warmup complete at step 100. Using CHRC only (SNMA disabled).
H-Mem Online (test): 100%|██████████████████████████████████████████████████████████████████▋| 42910/43105 [28:51<00:07, 25.13it/s, loss=0.000, mse=0.000, step=43000][W1220 20:51:33.000000000 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]
[W1220 20:51:33.000000000 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]

[H-Mem] TEST Results:
  MSE: 0.000000 | MAE: 0.000036 | RMSE: 0.000071
  Memory Bank: 1000 entries, 100.0% full
{'mae': [np.float32(3.589656e-05), np.float32(0.0)],
 'mse': [np.float32(5.0453486e-09), np.float32(0.0)]}


 2. python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma False --use_chrc True --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0
[H-Mem] Warmup complete at step 100. Using CHRC only (SNMA disabled).
H-Mem Online (test): 100%|██████████████████████████████████████████████████████████████████▊| 42990/43105 [28:48<00:04, 25.58it/s, loss=0.000, mse=0.000, step=43000][W1220 20:51:55.000000000 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]
[W1220 20:51:55.000000000 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]

[H-Mem] TEST Results:
  MSE: 0.000000 | MAE: 0.000036 | RMSE: 0.000071
  Memory Bank: 1000 entries, 100.0% full
{'mae': [np.float32(3.589656e-05), np.float32(0.0)],
 'mse': [np.float32(5.0453486e-09), np.float32(0.0)]}

 3. python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma False --use_chrc True --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.7 --chrc_gate_steepness 10.0
[H-Mem] Warmup complete at step 100. Using CHRC only (SNMA disabled).
H-Mem Online (test): 100%|██████████████████████████████████████████████████████████████████▋| 42922/43105 [28:39<00:07, 25.62it/s, loss=0.000, mse=0.000, step=43000][W1220 20:51:56.000000000 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]
[W1220 20:51:56.000000000 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]

[H-Mem] TEST Results:
  MSE: 0.000000 | MAE: 0.000036 | RMSE: 0.000071
  Memory Bank: 1000 entries, 100.0% full
{'mae': [np.float32(3.589656e-05), np.float32(0.0)],
 'mse': [np.float32(5.0453486e-09), np.float32(0.0)]}


4. python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation weighted_mean --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.3 --chrc_gate_steepness 10.0
H-Mem Online (test):   0%|                                                                                                      | 1/39427 [00:10<116:50:10, 10.67s/it]Traceback (most recent call last):
  File "E:\anaconda\envs\cl\lib\multiprocessing\queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
  File "E:\anaconda\envs\cl\lib\multiprocessing\reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
MemoryError
Traceback (most recent call last):
  File "E:\wyl\HMEM\run.py", line 533, in <module>
    mse, mae, test_data = exp.online(test_data, show_progress=True)
  File "E:\wyl\HMEM\exp\exp_hmem.py", line 474, in online
    for i, (recent_batch, current_batch) in enumerate(pbar):
  File "E:\anaconda\envs\cl\lib\site-packages\tqdm\std.py", line 1181, in __iter__
    for obj in iterable:
  File "E:\anaconda\envs\cl\lib\site-packages\torch\utils\data\dataloader.py", line 733, in __next__
    data = self._next_data()
  File "E:\anaconda\envs\cl\lib\site-packages\torch\utils\data\dataloader.py", line 1491, in _next_data
    idx, data = self._get_data()
  File "E:\anaconda\envs\cl\lib\site-packages\torch\utils\data\dataloader.py", line 1453, in _get_data
    success, data = self._try_get_data()
  File "E:\anaconda\envs\cl\lib\site-packages\torch\utils\data\dataloader.py", line 1284, in _try_get_data
    data = self._data_queue.get(timeout=timeout)
  File "E:\anaconda\envs\cl\lib\multiprocessing\queues.py", line 113, in get
    if not self._poll(timeout):
  File "E:\anaconda\envs\cl\lib\multiprocessing\connection.py", line 257, in poll
    return self._poll(timeout)
  File "E:\anaconda\envs\cl\lib\multiprocessing\connection.py", line 330, in _poll
    return bool(wait([self], timeout))
  File "E:\anaconda\envs\cl\lib\multiprocessing\connection.py", line 879, in wait
    ready_handles = _exhaustive_wait(waithandle_to_obj.keys(), timeout)
  File "E:\anaconda\envs\cl\lib\multiprocessing\connection.py", line 811, in _exhaustive_wait
    res = _winapi.WaitForMultipleObjects(L, False, timeout)
KeyboardInterrupt
[W1220 20:57:35.000000000 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]

5. python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation weighted_mean --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0
[H-Mem] Warmup complete at step 100. Using CHRC only (SNMA disabled).
H-Mem Online (test):  11%|███████▌                                                             | 4325/39427 [03:34<28:06, 20.81it/s, loss=0.000, mse=0.000, step=4000][W1220 20:28:12.000000000 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]
[W1220 20:28:12.000000000 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]
Traceback (most recent call last):
  File "E:\wyl\HMEM\run.py", line 533, in <module>
    mse, mae, test_data = exp.online(test_data, show_progress=True)
  File "E:\wyl\HMEM\exp\exp_hmem.py", line 543, in online
    running_mse = np.mean((recent_preds - recent_trues) ** 2)
numpy._core._exceptions._ArrayMemoryError: Unable to allocate 3.85 MiB for an array with shape (500, 96, 21) and data type float32

6. python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation weighted_mean --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.7 --chrc_gate_steepness 10.0
[H-Mem] Warmup complete at step 100. Using CHRC only (SNMA disabled).
H-Mem Online (test): 100%|██████████████████████████████████████████████████████████████████▊| 39344/39427 [30:38<00:04, 19.07it/s, loss=0.000, mse=0.000, step=39000][W1220 20:55:23.000000000 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]
[W1220 20:55:23.000000000 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]

[H-Mem] TEST Results:
  MSE: 0.000000 | MAE: 0.000091 | RMSE: 0.000190
  Memory Bank: 1000 entries, 100.0% full
{'mae': [np.float32(9.148616e-05), np.float32(0.0)],
 'mse': [np.float32(3.5919204e-08), np.float32(0.0)]}