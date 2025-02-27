#!/bin/bash
mvtec_ad_categories=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
visa_categories=('candle' 'capsules' 'cashew' 'chewinggum' 'fryum' 'macaroni1' 'macaroni2' 'pcb1' 'pcb2' 'pcb3' 'pcb4' 'pipe_fryum')

write_scores='benchmark.csv'
gpu_type='cuda'
gpu_number=0
num_workers=8

dataset_path="Data"
backbone='resnet18'
im_normalizer=True
pool=True
interpolation_mode='bilinear'
window_size=5
max_samples=750

for category in "${mvtec_ad_categories[@]}"; do
    python main.py --dataset mvtec_ad --dataset_path "$dataset_path" --category "$category" --backbone "$backbone" --gpu_type "$gpu_type" --gpu_number "$gpu_number" --num_workers "$num_workers" --write_scores "$write_scores" --pool "$pool" --interpolation_mode "$interpolation_mode" --image_normalization "$im_normalizer" --window_size "$window_size" --limit_train_samples "$max_samples"
done

for category in "${visa_categories[@]}"; do
    python main.py --dataset visa --dataset_path "$dataset_path" --category "$category" --preserve_aspect_ratio True --backbone "$backbone" --gpu_type "$gpu_type" --gpu_number "$gpu_number" --num_workers "$num_workers" --write_scores "$write_scores" --pool "$pool" --interpolation_mode "$interpolation_mode" --image_normalization "$im_normalizer" --window_size "$window_size" --limit_train_samples "$max_samples"
done
