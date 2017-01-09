#!/bin/sh
sh /home/fish.hy/tf_cluster/bin/start_cluster.sh
ssh tf2 "/home/fish.hy/tf_cluster/bin/start_cluster.sh"
ssh tf3 "/home/fish.hy/tf_cluster/bin/start_cluster.sh"
