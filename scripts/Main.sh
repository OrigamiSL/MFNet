python -u main.py --data PEMS03 --input_len 2016  --pred_len 96,192,336,720 --encoder_layer 3 --patch_size 24 --d_model 128 --U_num 0.1 --S_num 7 --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 10 --itr 5 --train --patience 3 --aug_p 0.5 --aug_e 0.5 --decay 0.5

python -u main.py --data PEMS04 --input_len 2016  --pred_len 96,192,336,720 --encoder_layer 3 --patch_size 24 --d_model 128 --U_num 0.1 --S_num 7 --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 10 --itr 5 --train --patience 3 --aug_p 0.5 --aug_e 0.5 --decay 0.5

python -u main.py --data PEMS07 --input_len 2016  --pred_len 96,192,336,720 --encoder_layer 3 --patch_size 24 --d_model 128 --U_num 0.1 --S_num 7 --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 10 --itr 5 --train --patience 3 --aug_p 0.5 --aug_e 0.5 --decay 0.5

python -u main.py --data PEMS08 --input_len 2016  --pred_len 96,192,336,720 --encoder_layer 3 --patch_size 24 --d_model 128 --U_num 0.1 --S_num 7 --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 10 --itr 5 --train --patience 3 --aug_p 0.5 --aug_e 0.5 --decay 0.5

python -u main.py --data CA-D5 --input_len 2016  --pred_len 96,192,336,720 --encoder_layer 3 --patch_size 24 --d_model 128 --U_num 0.1 --S_num 7 --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 10 --itr 5 --train --patience 3 --aug_p 0.5 --aug_e 0.5 --decay 0.5

python -u main.py --data Traffic --input_len 168  --pred_len 96,192,336,720 --encoder_layer 3 --patch_size 6 --d_model 128 --U_num 0.1 --S_num 7 --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 10 --itr 5 --train --patience 3 --aug_p 0.5 --aug_e 0.5 --decay 0.5
