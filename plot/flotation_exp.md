# Flotation

### pretrain
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 2 --train_epochs 10 --batch_size 32 --learning_rate 0.0001 --patience 3 --itr 1

python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 10 --train_epochs 10 --batch_size 32 --learning_rate 0.0001 --patience 3 --itr 1

python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 24 --train_epochs 10 --batch_size 32 --learning_rate 0.0001 --patience 3 --itr 1


### pred_len_2

#### frozen
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 2 --itr 1 --only_test --pretrain --wo_valid
#### online
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 2 --itr 1 --only_test --pretrain --wo_valid --online_method Online --online_learning_rate 1e-5
#### er
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 2 --itr 1 --only_test --pretrain --wo_valid --online_method ER --online_learning_rate 1e-5
#### derpp
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 2 --itr 1 --only_test --pretrain --wo_valid --online_method DERpp --online_learning_rate 1e-5
#### acl
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 2 --itr 1 --only_test --pretrain --wo_valid --online_method ACL --online_learning_rate 1e-5
#### clser
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 2 --itr 1 --only_test --pretrain --wo_valid --online_method CLSER --online_learning_rate 1e-5
#### mir
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 2 --itr 1 --only_test --pretrain --wo_valid --online_method MIR --online_learning_rate 1e-5
#### solid
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 2 --itr 1 --only_test --pretrain --wo_valid --online_method SOLID --online_learning_rate 1e-5
#### hmem
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 2 --itr 1 --only_test --pretrain --wo_valid --online_method HMem --online_learning_rate 1e-5 --use_chrc True --retrieval_top_k 5 --chrc_aggregation softmax --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_use_horizon_mask True --chrc_horizon_mask_mode exp --chrc_horizon_mask_decay 0.98 --chrc_horizon_mask_min 0.2 --chrc_use_buckets True --chrc_bucket_num 4

### pred_len_10

#### frozen
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 10 --itr 1 --only_test --pretrain --wo_valid
#### online
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 10 --itr 1 --only_test --pretrain --wo_valid --online_method Online --online_learning_rate 1e-5
#### er
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 10 --itr 1 --only_test --pretrain --wo_valid --online_method ER --online_learning_rate 1e-5
#### derpp
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 10 --itr 1 --only_test --pretrain --wo_valid --online_method DERpp --online_learning_rate 1e-5
#### acl
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 10 --itr 1 --only_test --pretrain --wo_valid --online_method ACL --online_learning_rate 1e-5
#### clser
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 10 --itr 1 --only_test --pretrain --wo_valid --online_method CLSER --online_learning_rate 1e-5
#### mir
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 10 --itr 1 --only_test --pretrain --wo_valid --online_method MIR --online_learning_rate 1e-5
#### solid
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 10 --itr 1 --only_test --pretrain --wo_valid --online_method SOLID --online_learning_rate 1e-5
#### hmem
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 10 --itr 1 --only_test --pretrain --wo_valid --online_method HMem --online_learning_rate 1e-5 --use_chrc True --retrieval_top_k 5 --chrc_aggregation softmax --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_use_horizon_mask True --chrc_horizon_mask_mode exp --chrc_horizon_mask_decay 0.98 --chrc_horizon_mask_min 0.2 --chrc_use_buckets True --chrc_bucket_num 4

### pred_len_24

#### frozen
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 24 --itr 1 --only_test --pretrain --wo_valid
#### online
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 24 --itr 1 --only_test --pretrain --wo_valid --online_method Online --online_learning_rate 1e-5
#### er
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 24 --itr 1 --only_test --pretrain --wo_valid --online_method ER --online_learning_rate 1e-5
#### derpp
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 24 --itr 1 --only_test --pretrain --wo_valid --online_method DERpp --online_learning_rate 1e-5
#### acl
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 24 --itr 1 --only_test --pretrain --wo_valid --online_method ACL --online_learning_rate 1e-5
#### clser
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 24 --itr 1 --only_test --pretrain --wo_valid --online_method CLSER --online_learning_rate 1e-5
#### mir
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 24 --itr 1 --only_test --pretrain --wo_valid --online_method MIR --online_learning_rate 1e-5
#### solid
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 24 --itr 1 --only_test --pretrain --wo_valid --online_method SOLID --online_learning_rate 1e-5
#### hmem
python -u run.py --dataset Flotation --border_type online --model iTransformer --seq_len 64 --pred_len 24 --itr 1 --only_test --pretrain --wo_valid --online_method HMem --online_learning_rate 1e-5 --use_chrc True --retrieval_top_k 5 --chrc_aggregation softmax --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_use_horizon_mask True --chrc_horizon_mask_mode exp --chrc_horizon_mask_decay 0.98 --chrc_horizon_mask_min 0.2 --chrc_use_buckets True --chrc_bucket_num 4
