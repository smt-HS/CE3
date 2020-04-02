#!/usr/bin/env bash

run_list(){
    cuda=$1
    topics="${@:2}"
    for i in ${topics};
    do
        CUDA_VISIBLE_DEVICES=${cuda} python run_v10.py --num_timesteps=3e6 --env=ds-v102 --topic=dd17-${i} --num_env=4 --save_path=model/dd17-${i}.model >ds_log/dd17-${i}.log 2>&1
    done

}


run_list 1 `seq 1 10` &
run_list 1 `seq 11 20` &
run_list 2 `seq 21 30` &
run_list 2 `seq 31 40` &
run_list 3 `seq 41 50` &
run_list 3 `seq 51 60` &
