#!/bin/sh
python grpc_tensorflow_server.py --cluster_spec 'worker|localhost:2222,ps|localhost:2223;localhost:2224' --job_name worker --task_id 0 &
python grpc_tensorflow_server.py --cluster_spec 'worker|localhost:2222,ps|localhost:2223;localhost:2224' --job_name ps --task_id 0 &
python grpc_tensorflow_server.py --cluster_spec 'worker|localhost:2222,ps|localhost:2223;localhost:2224' --job_name ps --task_id 1 &
