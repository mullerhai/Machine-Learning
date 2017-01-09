#!/bin/sh
servers=`ps -ef | grep grpc | grep python | grep fish.hy | awk '{print $2}'`
for s in $servers
do
    kill -9 $s
    kill -9 $s
done
#python ./grpc_tensorflow_server.py --cluster_spec 'worker|tf1.kgb.et2:3222;tf2.kgb.et2:3222;tf3.kgb.et2:3222,ps|tf1.kgb.et2:3224' --job_name worker --task_id 0 &
#python ./grpc_tensorflow_server.py --cluster_spec 'worker|tf1.kgb.et2:3222;tf2.kgb.et2:3222;tf3.kgb.et2:3222,ps|tf1.kgb.et2:3224' --job_name worker --task_id 1 &
#python ./grpc_tensorflow_server.py --cluster_spec 'worker|tf1.kgb.et2:3222;tf2.kgb.et2:3222;tf3.kgb.et2:3222,ps|tf1.kgb.et2:3224' --job_name worker --task_id 2 &
#python ./grpc_tensorflow_server.py --cluster_spec 'worker|tf1.kgb.et2:3222;tf2.kgb.et2:3222;tf3.kgb.et2:3222,ps|tf1.kgb.et2:3224' --job_name ps --task_id 0 &
