#!/bin/bash
mvtec_ad_categories=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
visa_categories=('candle' 'capsules' 'cashew' 'chewinggum' 'fryum' 'macaroni1' 'macaroni2' 'pcb1' 'pcb2' 'pcb3' 'pcb4' 'pipe_fryum')

write_scores='ablation_study.csv'
gpu_type='cuda'
gpu_number=0
num_workers=8
dataset_path="Data"

backbone='resnet18'

im_normalizers=(False True)
pools=(False True)
interpolation_modes=('nearest' 'bilinear')
window_sizes=(1 5)
max_samples=750
aspect_ratios=(False True)
for pool in "${pools[@]}"; do
    for window_size in "${window_sizes[@]}"; do
        for interpolation_mode in "${interpolation_modes[@]}"; do
            for im_normalizer in "${im_normalizers[@]}"; do
                for category in "${mvtec_ad_categories[@]}"; do
                    python main.py --dataset mvtec_ad --category "$category" --dataset_path "$dataset_path" --backbone "$backbone" --gpu_type "$gpu_type" --gpu_number "$gpu_number" --num_workers "$num_workers" --write_scores "$write_scores" --pool "$pool" --interpolation_mode "$interpolation_mode" --image_normalization "$im_normalizer" --window_size "$window_size" --train_samples "$max_samples"
                done

                for aspect_ratio in "${aspect_ratios[@]}"; do
                    for category in "${visa_categories[@]}"; do
                        python main.py --dataset visa --category "$category" --dataset_path "$dataset_path" --backbone "$backbone" --gpu_type "$gpu_type" --gpu_number "$gpu_number" --num_workers "$num_workers" --write_scores "$write_scores" --pool "$pool" --interpolation_mode "$interpolation_mode" --image_normalization "$im_normalizer" --preserve_aspect_ratio "$aspect_ratio" --window_size "$window_size" --train_samples "$max_samples"
                    done
                done

            done
        done
    done
done
