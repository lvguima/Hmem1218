# ETTM1

### pretrain
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 24 --train_epochs 10 --batch_size 32 --learning_rate 0.0001 --patience 3 --itr 1

python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 48 --train_epochs 10 --batch_size 32 --learning_rate 0.0001 --patience 3 --itr 1

python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --train_epochs 10 --batch_size 32 --learning_rate 0.0001 --patience 3 --itr 1


### pred_len_24

#### frozen
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 24 --itr 1 --only_test --pretrain
MSE:0.609208, MAE:0.479074, RMSE:0.780518, RSE:0.485051, R2:0.764725, MAPE:2.254775
#### online
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 24 --itr 1 --only_test --pretrain --online_method Online --online_learning_rate 1e-5
MSE:0.614837, MAE:0.488748, RMSE:0.784116, RSE:0.487287, R2:0.762551, MAPE:2.316432
#### er
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 24 --itr 1 --only_test --pretrain --online_method ER --online_learning_rate 1e-5
MSE:0.549841, MAE:0.457814, RMSE:0.741513, RSE:0.460812, R2:0.787653, MAPE:2.230724
#### derpp
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 24 --itr 1 --only_test --pretrain --online_method DERpp --online_learning_rate 1e-5
MSE:0.550050, MAE:0.456358, RMSE:0.741653, RSE:0.460899, R2:0.787572, MAPE:2.200759
#### acl
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 24 --itr 1 --only_test --pretrain --online_method ACL --online_learning_rate 1e-5 
MSE:0.557351, MAE:0.460279, RMSE:0.746559, RSE:0.463948, R2:0.784752, MAPE:2.240259
#### clser
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 24 --itr 1 --only_test --pretrain --online_method CLSER --online_learning_rate 1e-5 
MSE:0.570717, MAE:0.467961, RMSE:0.755458, RSE:0.469478, R2:0.779590, MAPE:2.252246
#### mir
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 24 --itr 1 --only_test --pretrain --online_method MIR --online_learning_rate 1e-5 
MSE:0.492129, MAE:0.430472, RMSE:0.701519, RSE:0.435958, R2:0.809941, MAPE:2.088288
#### solid
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 24 --itr 1 --only_test --pretrain --online_method SOLID --online_learning_rate 1e-5 
MSE:0.609116, MAE:0.479043, RMSE:0.780459, RSE:0.485015, R2:0.764761, MAPE:2.254615
#### hmem
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 24 --itr 1 --only_test --pretrain --online_method HMem --online_learning_rate 1e-5 --use_chrc True --retrieval_top_k 5 --chrc_aggregation softmax --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_use_horizon_mask True --chrc_horizon_mask_mode exp --chrc_horizon_mask_decay 0.98 --chrc_horizon_mask_min 0.2 --chrc_use_buckets True --chrc_bucket_num 4 
MSE: 0.481261 | MAE: 0.427727 | RMSE: 0.693730 | RSE: 0.431117 | R2: 0.814138 | MAPE: 2.077712

### pred_len_48

#### frozen
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 48 --itr 1 --only_test --pretrain
MSE:0.797007, MAE:0.563564, RMSE:0.892753, RSE:0.554836, R2:0.692157, MAPE:2.698190
#### online
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 48 --itr 1 --only_test --pretrain --online_method Online --online_learning_rate 1e-5
MSE:0.773044, MAE:0.551876, RMSE:0.879229, RSE:0.546432, R2:0.701412, MAPE:2.604756
#### er
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 48 --itr 1 --only_test --pretrain --online_method ER --online_learning_rate 1e-5
MSE:0.701984, MAE:0.520859, RMSE:0.837845, RSE:0.520712, R2:0.728859, MAPE:2.469954
#### derpp
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 48 --itr 1 --only_test --pretrain --online_method DERpp --online_learning_rate 1e-5
MSE:0.704616, MAE:0.521971, RMSE:0.839414, RSE:0.521687, R2:0.727843, MAPE:2.468185
#### acl
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 48 --itr 1 --only_test --pretrain --online_method ACL --online_learning_rate 1e-5
MSE:0.709204, MAE:0.525131, RMSE:0.842143, RSE:0.523383, R2:0.726070, MAPE:2.501234
#### clser
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 48 --itr 1 --only_test --pretrain --online_method CLSER --online_learning_rate 1e-5
MSE:0.722427, MAE:0.530941, RMSE:0.849957, RSE:0.528239, R2:0.720963, MAPE:2.491102
#### mir
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 48 --itr 1 --only_test --pretrain --online_method MIR --online_learning_rate 1e-5
MSE:0.663831, MAE:0.505195, RMSE:0.814758, RSE:0.506364, R2:0.743596, MAPE:2.422400
#### solid
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 48 --itr 1 --only_test --pretrain --online_method SOLID --online_learning_rate 1e-5
MSE:0.796954, MAE:0.563548, RMSE:0.892723, RSE:0.554818, R2:0.692177, MAPE:2.698105
#### hmem
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 48 --itr 1 --only_test --pretrain --online_method HMem --online_learning_rate 1e-5 --use_chrc True --retrieval_top_k 5 --chrc_aggregation softmax --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_use_horizon_mask True --chrc_horizon_mask_mode exp --chrc_horizon_mask_decay 0.98 --chrc_horizon_mask_min 0.2 --chrc_use_buckets True --chrc_bucket_num 4
MSE: 0.644673 | MAE: 0.508949 | RMSE: 0.802915 | RSE: 0.499003 | R2: 0.750996 | MAPE: 2.488347

### pred_len_96

#### frozen
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --only_test --pretrain
MSE:0.852166, MAE:0.590607, RMSE:0.923128, RSE:0.573612, R2:0.670969, MAPE:2.865086
#### online
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --only_test --pretrain --online_method Online --online_learning_rate 1e-5
MSE:0.915850, MAE:0.603770, RMSE:0.957000, RSE:0.594659, R2:0.646381, MAPE:2.801484
#### er
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --only_test --pretrain --online_method ER --online_learning_rate 1e-5
MSE:0.836412, MAE:0.576477, RMSE:0.914555, RSE:0.568285, R2:0.677052, MAPE:2.689655
#### derpp
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --only_test --pretrain --online_method DERpp --online_learning_rate 1e-5
MSE:0.829074, MAE:0.572346, RMSE:0.910535, RSE:0.565787, R2:0.679886, MAPE:2.676994
#### acl
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --only_test --pretrain --online_method ACL --online_learning_rate 1e-5
MSE:0.839617, MAE:0.578122, RMSE:0.916306, RSE:0.569373, R2:0.675815, MAPE:2.682764
#### clser
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --only_test --pretrain --online_method CLSER --online_learning_rate 1e-5
MSE:0.855892, MAE:0.583966, RMSE:0.925144, RSE:0.574865, R2:0.669531, MAPE:2.720927
#### mir
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --only_test --pretrain --online_method MIR --online_learning_rate 1e-5
MSE:0.756556, MAE:0.548490, RMSE:0.869802, RSE:0.540476, R2:0.707885, MAPE:2.562611
#### solid
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --only_test --pretrain --online_method SOLID --online_learning_rate 1e-5
MSE:0.852132, MAE:0.590596, RMSE:0.923110, RSE:0.573600, R2:0.670983, MAPE:2.865012
#### hmem
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --only_test --pretrain --online_method HMem --online_learning_rate 1e-5 --use_chrc True --retrieval_top_k 5 --chrc_aggregation softmax --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_use_horizon_mask True --chrc_horizon_mask_mode exp --chrc_horizon_mask_decay 0.98 --chrc_horizon_mask_min 0.2 --chrc_use_buckets True --chrc_bucket_num 4
MSE: 0.708351 | MAE: 0.535843 | RMSE: 0.841636 | RSE: 0.522974 | R2: 0.726498 | MAPE: 2.585476