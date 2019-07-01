python -m gs_baseline.supervised_train --data_prefix ./data.ignore/data.ignore/$1 --model graphsage_mean --epochs $2 --samples_1 $3 --samples_2 $4 --dim_1 256 --dim_2 256 --sigmoid
