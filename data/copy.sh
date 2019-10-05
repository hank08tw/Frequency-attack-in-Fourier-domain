#!/bin/bash
num=100
for ((i=2;i<=num;i++))
do
     cp "./test_data/1.jpeg" "./test_data/${i}.jpeg"
done
