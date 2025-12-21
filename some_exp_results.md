
### P2实验结果

`
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma True --use_chrc False --lora_ema_decay 0.9  MSE: 0.775903 | MAE: 0.558986 | RMSE: 0.880854
`

`python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma False --use_chrc True --lora_ema_decay 0.9 --chrc_min_similarity 0.6   MSE: 0.780064 | MAE: 0.563025 | RMSE: 0.883212
`

`
 python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma True --use_chrc True --lora_ema_decay 0.9 --chrc_min_similarity 0.6
 MSE: 0.813580 | MAE: 0.573263 | RMSE: 0.901987
`

`
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma True --use_chrc False --lora_ema_decay 0.9
 MSE: 0.807573 | MAE: 0.571548 | RMSE: 0.898651
`

`
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma True --use_chrc False --lora_ema_decay 0.9 --chrc_aggregation weighted_mean
MSE: 2.316577 | MAE: 0.863359 | RMSE: 1.52203
`

`
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma True --use_chrc True --lora_ema_decay 0.9 --chrc_min_similarity 0.6
MSE: 0.784931 | MAE: 0.566108 | RMSE: 0.885963
`

`
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 3e-5 --use_snma True --use_chrc True --lora_ema_decay 0.9 --chrc_min_similarity 0.6
MSE: 0.914286 | MAE: 0.610256 | RMSE: 0.956183
`

`
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma True --use_chrc True --lora_ema_decay 0.95 --chrc_min_similarity 0.6
 MSE: 0.810633 | MAE: 0.571947 | RMSE: 0.900352
`

`
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma True --use_chrc True --lora_ema_decay 0.8 --chrc_min_similarity 0.6
MSE: 0.810633 | MAE: 0.571947 | RMSE: 0.900352
`

`
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma False --use_chrc False --freeze True
 MSE: 0.829308 | MAE: 0.583861 | RMSE: 0.910664
`

`
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc False --freeze True
 MSE: 1.714104 | MAE: 0.734445 | RMSE: 1.309238
`

ETTm1 三个chrc_min_similarity的结果 0.3:  MSE: 0.810633 | MAE: 0.571947 | RMSE: 0.90035, 0.6:MSE: 0.810633 | MAE: 0.571947 | RMSE: 0.900352 0.9:MSE: 0.810633 | MAE: 0.571947 | RMSE: 0.900352 Weather 三个chrc_min_similarity的结果 0.3:MSE: 1.577724 | MAE: 0.716426 | RMSE: 1.256075 0.6:MSE: 1.577724 | MAE: 0.716426 | RMSE: 1.256075 0.9: MSE: 1.577724 | MAE: 0.716426 | RMSE: 1.256075