#!/bin/bash
cd /data1/tongjizhou/FedAgent
export CUDA_VISIBLE_DEVICES=3
/data1/tongjizhou/miniconda3/envs/realm/bin/python scripts/e24_7b_validation.py > outputs/e24_7b_validation/e24.log 2>&1
