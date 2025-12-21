### P4实验结果

 python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma False --use_chrc True --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_trajectory_bias 0.0 --chrc_aggregation softmax
MSE: 0.755514 | MAE: 0.555819 | RMSE: 0.869203

python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma 
False --use_chrc True --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_trajectory_bias 0.0 --chrc_aggregation adaptive
MSE: 0.772021 | MAE: 0.559702 | RMSE: 0.878647


python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation weighted_mean --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_trajectory_bias 0.0
MSE: 1.577724 | MAE: 0.716426 | RMSE: 1.256075

python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation adaptive --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_trajectory_bias 0.0
MSE: 1.944069 | MAE: 0.826229 | RMSE: 1.394299