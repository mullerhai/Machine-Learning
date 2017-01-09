#!/bin/sh
#servers=`ps -ef | grep grpc | grep python | grep fish.hy | awk '{print $2}'`
#for s in $servers
#do
#    kill -9 $s
#    kill -9 $s
#done
sh /home/fish.hy/tf_cluster/bin/stop_cluster.sh
ssh tf2 "sh /home/fish.hy/tf_cluster/bin/stop_cluster.sh"
ssh tf3 "sh /home/fish.hy/tf_cluster/bin/stop_cluster.sh"
