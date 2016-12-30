#!/bin/bash
nohup python ./mnist_replica.py --cluster_spec 'worker|10.98.52.91:2222;10.98.20.102:2222,ps|10.98.8.30:2224' --train_steps 1000 --sync_replicas 1 --replicas_to_aggregate 2 --worker_index 0 --num_workers 2 --num_parameter_servers 1 --worker_grpc_url grpc://10.98.52.91:2222 > log0 2>&1 &
nohup python ./mnist_replica.py --cluster_spec 'worker|10.98.52.91:2222;10.98.20.102:2222,ps|10.98.8.30:2224' --train_steps 1000 --sync_replicas 1 --replicas_to_aggregate 2 --worker_index 1 --num_workers 2 --num_parameter_servers 1 --worker_grpc_url grpc://10.98.20.102:2222 > log1 2>&1 &
