#!/bin/bash
for((i=10;i<30;i++))
do

for((j=1;j<20;j++))
do
./light-svm  -train test/data/heart_scale.train  -model svm_model -valid test/data/heart_scale.test  -c $i -sigma $j  >> result
done

done
