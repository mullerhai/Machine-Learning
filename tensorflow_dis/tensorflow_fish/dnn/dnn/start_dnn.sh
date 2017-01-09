#!/usr/bin/bash
python dnn_dis.py --data_dir /home/chuanxin.tcx/Test/Data/ --train_dir log/ --num_workers 3 --num_parameter_servers 1 --worker_grpc_url grpc://tf1.kgb.et2:3222
