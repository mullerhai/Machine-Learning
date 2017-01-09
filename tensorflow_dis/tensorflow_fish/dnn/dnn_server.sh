#!/bin/sh
python ./grpc_tensorflow_server.py --cluster_spec 'worker|tf1.kgb.et2:3222;tf2.kgb.et2:3222;tf3.kgb.et2:3222,ps|tf1.kgb.et2:3224' --job_name worker --task_id 0 &
python ./grpc_tensorflow_server.py --cluster_spec 'worker|tf1.kgb.et2:3222;tf2.kgb.et2:3222;tf3.kgb.et2:3222,ps|tf1.kgb.et2:3224' --job_name worker --task_id 1 &
python ./grpc_tensorflow_server.py --cluster_spec 'worker|tf1.kgb.et2:3222;tf2.kgb.et2:3222;tf3.kgb.et2:3222,ps|tf1.kgb.et2:3224' --job_name worker --task_id 2 &
python ./grpc_tensorflow_server.py --cluster_spec 'worker|tf1.kgb.et2:3222;tf2.kgb.et2:3222;tf3.kgb.et2:3222,ps|tf1.kgb.et2:3224' --job_name ps --task_id 0 &
