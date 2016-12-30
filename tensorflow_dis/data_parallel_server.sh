#!/bin/sh
./grpc_tensorflow_server.py --cluster_spec 'master|localhost:2222,worker|localhost:2223,worker_|localhost:2224' --job_name master --task_id 0 &
./grpc_tensorflow_server.py --cluster_spec 'master|localhost:2222,worker|localhost:2223,worker_|localhost:2224' --job_name worker --task_id 0 &
./grpc_tensorflow_server.py --cluster_spec 'master|localhost:2222,worker|localhost:2223,worker_|localhost:2224' --job_name worker_ --task_id 0 &
