# Weather

### pretrain
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 24 --train_epochs 10 --batch_size 32 --learning_rate 0.0001 --patience 3 --itr 1

python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 48 --train_epochs 10 --batch_size 32 --learning_rate 0.0001 --patience 3 --itr 1

python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --train_epochs 10 --batch_size 32 --learning_rate 0.0001 --patience 3 --itr 1


### pred_len_24

#### frozen
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 24 --itr 1 --only_test --pretrain --wo_valid
MSE:0.998398, MAE:0.493111, RMSE:0.999199, RSE:0.405629, R2:0.835465, MAPE:1.675203
#### online
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 24 --itr 1 --only_test --pretrain --wo_valid --online_method Online --online_learning_rate 1e-5
MSE:1.647988, MAE:0.713391, RMSE:1.283740, RSE:0.521140, R2:0.728413, MAPE:2.507346
#### er
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 24 --itr 1 --only_test --pretrain --wo_valid --online_method ER --online_learning_rate 1e-5
MSE:0.961833, MAE:0.494037, RMSE:0.980731, RSE:0.398132, R2:0.841491, MAPE:1.773008
#### derpp
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 24 --itr 1 --only_test --pretrain --wo_valid --online_method DERpp --online_learning_rate 1e-5
MSE:0.951673, MAE:0.491511, RMSE:0.975537, RSE:0.396024, R2:0.843165, MAPE:1.741937
#### acl
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 24 --itr 1 --only_test --pretrain --wo_valid --online_method ACL --online_learning_rate 1e-5
MSE:0.980501, MAE:0.500303, RMSE:0.990202, RSE:0.401977, R2:0.838414, MAPE:1.785084
#### clser
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 24 --itr 1 --only_test --pretrain --wo_valid --online_method CLSER --online_learning_rate 1e-5
MSE:1.225008, MAE:0.574597, RMSE:1.106801, RSE:0.449311, R2:0.798120, MAPE:2.049331
#### mir
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 24 --itr 1 --only_test --pretrain --wo_valid --online_method MIR --online_learning_rate 1e-5
MSE: 0.902783 | MAE: 0.474836 | RMSE: 0.950149 | RSE: 0.385717 | R2: 0.851222 | MAPE: 2.100387
#### solid
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 24 --itr 1 --only_test --pretrain --wo_valid --online_method SOLID --online_learning_rate 1e-5
MSE:0.998066, MAE:0.493006, RMSE:0.999033, RSE:0.405562, R2:0.835520, MAPE:1.675003
#### hmem
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 24 --itr 1 --only_test --pretrain --wo_valid --online_method HMem --online_learning_rate 1e-5 --use_chrc True --retrieval_top_k 5 --chrc_aggregation softmax --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_use_horizon_mask True --chrc_horizon_mask_mode exp --chrc_horizon_mask_decay 0.98 --chrc_horizon_mask_min 0.2 --chrc_use_buckets True --chrc_bucket_num 4
MSE:0.861584, MAE:0.449596, RMSE:0.928215, RSE:0.376813, R2:0.858012, MAPE:1.599082

### pred_len_48

#### frozen
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 48 --itr 1 --only_test --pretrain --wo_valid
MSE:1.372492, MAE:0.620773, RMSE:1.171534, RSE:0.475538, R2:0.773863, MAPE:2.087227
#### online
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 48 --itr 1 --only_test --pretrain --wo_valid --online_method Online --online_learning_rate 1e-5
MSE:2.650410, MAE:0.925653, RMSE:1.628008, RSE:0.660826, R2:0.563309, MAPE:3.322276
#### er
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 48 --itr 1 --only_test --pretrain --wo_valid --online_method ER --online_learning_rate 1e-5
MSE:1.329095, MAE:0.619577, RMSE:1.152864, RSE:0.467960, R2:0.781014, MAPE:2.227610
#### derpp
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 48 --itr 1 --only_test --pretrain --wo_valid --online_method DERpp --online_learning_rate 1e-5
MSE:1.319630, MAE:0.617312, RMSE:1.148752, RSE:0.466291, R2:0.782573, MAPE:2.192843
#### acl
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 48 --itr 1 --only_test --pretrain --wo_valid --online_method ACL --online_learning_rate 1e-5
MSE:1.351062, MAE:0.627077, RMSE:1.162352, RSE:0.471811, R2:0.777394, MAPE:2.292248
#### clser
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 48 --itr 1 --only_test --pretrain --wo_valid --online_method CLSER --online_learning_rate 1e-5
MSE:1.568007, MAE:0.691834, RMSE:1.252201, RSE:0.508282, R2:0.741650, MAPE:2.524212
#### mir
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 48 --itr 1 --only_test --pretrain --wo_valid --online_method MIR --online_learning_rate 1e-5
MSE: 1.309058 | MAE: 0.612852 | RMSE: 1.144141 | RSE: 0.464419 | R2: 0.784315 | MAPE: 2.443073
#### solid
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 48 --itr 1 --only_test --pretrain --wo_valid --online_method SOLID --online_learning_rate 1e-5
MSE:1.372260, MAE:0.620715, RMSE:1.171435, RSE:0.475498, R2:0.773902, MAPE:2.087097
#### hmem
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 48 --itr 1 --only_test --pretrain --wo_valid --online_method HMem --online_learning_rate 1e-5 --use_chrc True --retrieval_top_k 5 --chrc_aggregation softmax --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_use_horizon_mask True --chrc_horizon_mask_mode exp --chrc_horizon_mask_decay 0.98 --chrc_horizon_mask_min 0.2 --chrc_use_buckets True --chrc_bucket_num 4
MSE:1.196430, MAE:0.569942, RMSE:1.093814, RSE:0.443991, R2:0.802872, MAPE:2.089463

### pred_len_96

#### frozen
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --only_test --pretrain --wo_valid
MSE:1.723650, MAE:0.737436, RMSE:1.312879, RSE:0.532765, R2:0.716161, MAPE:2.485849
#### online
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --only_test --pretrain --wo_valid --online_method Online --online_learning_rate 1e-5
MSE:3.568185, MAE:1.077196, RMSE:1.888964, RSE:0.766540, R2:0.412417, MAPE:3.756906
#### er
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --only_test --pretrain --wo_valid --online_method ER --online_learning_rate 1e-5
MSE:1.715719, MAE:0.742753, RMSE:1.309855, RSE:0.531538, R2:0.717468, MAPE:2.593451
#### derpp
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --only_test --pretrain --wo_valid --online_method DERpp --online_learning_rate 1e-5
MSE:1.791747, MAE:0.753724, RMSE:1.338561, RSE:0.543187, R2:0.704948, MAPE:2.624445
#### acl
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --only_test --pretrain --wo_valid --online_method ACL --online_learning_rate 1e-5
MSE:1.766038, MAE:0.752043, RMSE:1.328924, RSE:0.539276, R2:0.709181, MAPE:2.633718
#### clser
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --only_test --pretrain --wo_valid --online_method CLSER --online_learning_rate 1e-5
MSE:1.766038, MAE:0.752043, RMSE:1.328924, RSE:0.539276, R2:0.709181, MAPE:2.633718
#### mir
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --only_test --pretrain --wo_valid --online_method MIR --online_learning_rate 1e-5
MSE: 1.725630 | MAE: 0.742212 | RMSE: 1.313632 | RSE: 0.533071 | R2: 0.715836 | MAPE: 2.931843
#### solid
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --only_test --pretrain --wo_valid --online_method SOLID --online_learning_rate 1e-5
MSE:1.723528, MAE:0.737405, RMSE:1.312832, RSE:0.532746, R2:0.716182, MAPE:2.485780
#### hmem
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --only_test --pretrain --wo_valid --online_method HMem --online_learning_rate 1e-5 --use_chrc True --retrieval_top_k 5 --chrc_aggregation softmax --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_use_horizon_mask True --chrc_horizon_mask_mode exp --chrc_horizon_mask_decay 0.98 --chrc_horizon_mask_min 0.2 --chrc_use_buckets True --chrc_bucket_num 4
MSE:1.513829, MAE:0.681834, RMSE:1.230378, RSE:0.499286, R2:0.750713, MAPE:2.431107