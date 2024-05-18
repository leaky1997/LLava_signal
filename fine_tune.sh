# export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}} 
unset LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64{LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
xtuner train /home/user/LQ/B_Signal/LLava_signal/Llama3-XTuner-CN/configs/llama3-llava/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_lora_e1_finetune.py \
--work-dir work_dir/llava \
--deepspeed deepspeed_zero3