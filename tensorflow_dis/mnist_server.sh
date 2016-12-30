#!/bin/sh
python ./grpc_tensorflow_server.py --cluster_spec 'worker|10.98.52.91:2222;10.98.20.102:2222,ps|10.98.8.30:2224' --job_name worker --task_id 0 &
python ./grpc_tensorflow_server.py --cluster_spec 'worker|10.98.52.91:2222;10.98.20.102:2222,ps|10.98.8.30:2224' --job_name worker --task_id 1 &
python ./grpc_tensorflow_server.py --cluster_spec 'worker|10.98.52.91:2222;10.98.20.102:2222,ps|10.98.8.30:2224' --job_name ps --task_id 0 &
