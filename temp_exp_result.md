
### P2实验结果

1. python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma False --use_chrc True --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.3 --chrc_gate_steepness 10.0
MSE: 0.755514 | MAE: 0.555819 | RMSE: 0.869203


 2. python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma False --use_chrc True --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0
MSE: 0.755514 | MAE: 0.555819 | RMSE: 0.869203

 3. python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma False --use_chrc True --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.7 --chrc_gate_steepness 10.0
MSE: 0.755514 | MAE: 0.555819 | RMSE: 0.869203


1. python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation weighted_mean --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.3 --chrc_gate_steepness 10.0
MSE: 1.577724 | MAE: 0.716426 | RMSE: 1.256075

5. python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation weighted_mean --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0
MSE: 1.577724 | MAE: 0.716426 | RMSE: 1.256075


6. python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation weighted_mean --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.7 --chrc_gate_steepness 10.0
 MSE: 1.577724 | MAE: 0.716426 | RMSE: 1.256075



### P3实验结果

1. python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma False --use_chrc True --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_trajectory_bias 0.0
MSE: 0.772096 | MAE: 0.559695 | RMSE: 0.878690

2. python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma False --use_chrc True --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_trajectory_bias 0.2
MSE: 0.772702 | MAE: 0.559693 | RMSE: 0.879035

3. python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation weighted_mean --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_trajectory_bias 0.0
 MSE: 1.919845 | MAE: 0.818350 | RMSE: 1.385585

4. python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation weighted_mean --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_trajectory_bias 0.2
MSE: 1.924322 | MAE: 0.819975 | RMSE: 1.387199