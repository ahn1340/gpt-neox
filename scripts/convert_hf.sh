#!/bin/bash

# NeoX 체크포인트를 HF 체크포인트로 변환하는 스크립트
#TODO: do this automatically
python /admin/home-jinwooahn/repos/gpt-neox/tools/ckpts/convert_module_to_hf.py --input_dir /weka/home-jinwooahn/polyglot-v2/checkpoints/global_step5000 --config_file /admin/home-jinwooahn/repos/gpt-neox/configs/polyglot-v2/6-9B-convert.yml --output_dir /weka/home-jinwooahn/hf-checkpoints/step5000
