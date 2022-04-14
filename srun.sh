srun --gres=gpu:GTX1080Ti:1 --nodelist=chpc-gpu002 python -u train_gpt.py --coord_feature_dim 256 \
                --batch_size 90 \
                --num_boxes 30 \
                --dataset crosstask \
                --model global_i3d \
                --search_method beam \
                --gpt_repr one \
                --generation_method autoregression \
                --sample_eval_gpt True \
                --pred_state_action True \
                --lr 0.01
