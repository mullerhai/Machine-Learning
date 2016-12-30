#!/bin/sh
sudo kill -9 $(ps aux | grep grpc | awk '{print $2}')
python ./grpc_tensorflow_server.py --cluster_spec 'worker|10.103.28.106:2222;10.103.123.63:2222,ps|10.103.122.200:2224' --job_name worker --task_id 1
