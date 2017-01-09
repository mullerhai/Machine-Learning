#!/usr/bin/bash
#python dnn_dis_1002.py --data_dir /home/chuanxin.tcx/Test/Data/ --train_dir log/ --num_workers 3 --num_parameter_servers 1 --worker_grpc_url grpc://tf1.kgb.et2:3222
#python dnn_dis.py --data_dir data/ --train_dir log/ --num_workers 3 --num_parameter_servers 1 --worker_grpc_url grpc://tf1.kgb.et2:3222
#python dnn_model.py --data_dir /home/fish.hy/dnn/data/ --log_dir /home/fish.hy/dnn/log/ --num_parameter_servers 1 --master grpc://tf3.kgb.et2:3222 --report_file /home/fish.hy/dnn/report.log --worker_index 0
#python dnn_model.py --data_dir /home/fish.hy/dnn/data/ --log_dir /home/fish.hy/dnn/log/ --num_parameter_servers 1 --master grpc://tf3.kgb.et2:3222 --report_file /home/fish.hy/dnn/report.log --worker_index 0 --batch_size 100 --eval_iter 20 --eval_batch_size 50000 --eval_secs 180 --report_step 50
python dnn_model_auc.py --data_dir /home/fish.hy/dnn/data/ --log_dir /home/fish.hy/dnn/log/ --num_parameter_servers 1 --master grpc://tf3.kgb.et2:3222 --report_file /home/fish.hy/dnn/report.log --worker_index 1 --batch_size 2000 --eval_iter 10 --eval_batch_size 50000 --eval_secs 1200 --report_step 500 --optimizer adam --num_workers 3
