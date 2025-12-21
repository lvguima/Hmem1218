### P5实验结果
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma False --use_chrc True --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_trajectory_bias 0.0 --chrc_aggregation adaptive --chrc_use_error_decomp False
MSE: 0.758807 | MAE: 0.556801 | RMSE: 0.871096

python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma False --use_chrc True --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_trajectory_bias 0.0 --chrc_aggregation adaptive --chrc_use_error_decomp True --chrc_error_ema_decay 0.9
MSE: 0.758807 | MAE: 0.556801 | RMSE: 0.871096

python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation adaptive --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_trajectory_bias 0.0 --chrc_use_error_decomp False
MSE: 1.957755 | MAE: 0.832740 | RMSE: 1.399198


python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation adaptive --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_trajectory_bias 0.0 --chrc_use_error_decomp True --chrc_error_ema_decay 0.9
MSE: 1.957755 | MAE: 0.832740 | RMSE: 1.399198



### P6实验结果

python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma False --use_chrc True --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_trajectory_bias 0.0 --chrc_aggregation adaptive --chrc_use_context_key False
MSE: 0.772021 | MAE: 0.559702 | RMSE: 0.878647

python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma False --use_chrc True --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_trajectory_bias 0.0 --chrc_aggregation adaptive --chrc_use_context_key True --chrc_context_len 96
MSE: 0.771484 | MAE: 0.559225 | RMSE: 0.878341

python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation adaptive --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_trajectory_bias 0.0 --chrc_use_context_key False
MSE: 1.944069 | MAE: 0.826229 | RMSE: 1.394299

python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation adaptive --hmem_pogt_source batch_x --chrc_use_dual_key True --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_trajectory_bias 0.0 --chrc_use_context_key True --chrc_context_len 96
MSE: 1.944814 | MAE: 0.827698 | RMSE: 1.39456