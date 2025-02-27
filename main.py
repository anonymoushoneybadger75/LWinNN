import torch
import csv
from math import *
import time
import argparse
import os
from src.Dataloaders import MVTecAD, VisA
from src.LWinNN.LWinNN_Backend import lwinnn
import torchvision.transforms.v2 as transforms
from torcheval.metrics import BinaryAUROC

DATASETS = ['mvtec_ad', 'visa']
CATEGORIES = ['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper',
              'candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
BACKBONES = ['resnet18','wide_resnet50','wide_resnet101','resnet34','resnet50']
INTERPOLATION_MODES = ['nearest', 'bilinear']

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', choices=DATASETS, help='Parent dataset', type=str, default='mvtec_ad')
    parser.add_argument('--dataset_path', type=str, help='Path to parent folder of dataset', default='Data')
    parser.add_argument('--category', choices=CATEGORIES, help='category to test on', type=str, default='bottle')
    parser.add_argument('--backbone', choices=BACKBONES, help='Pretrained feature extractor', default='resnet18', type=str)
    
    parser.add_argument('--batch_size', help='batch size', default=32, type=int)
    parser.add_argument('--limit_train_samples', help='max number of train samples', default=-1, type=int)
    
    parser.add_argument('--gpu_type', type=str, default='cuda')
    parser.add_argument('--gpu_number', default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--write_scores', type=str, default='')

    parser.add_argument('--window_size', help='window size for nearest neighbor search', default=5,type=int)
    
    parser.add_argument('--image_normalization', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--preserve_aspect_ratio', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--pool', help='pool features', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--interpolation_mode', choices=INTERPOLATION_MODES, type=str, default='bilinear')

    args = parser.parse_args()

    if args.gpu_type != 'mps':
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_number)
    return args

def main(args):
    try:
        dataset = {'mvtec_ad':MVTecAD, 'visa':VisA}[args.dataset]
        train_set = dataset(args.dataset_path, category=args.category, train=True, normalize=args.image_normalization, preserve_aspect_ratio=args.preserve_aspect_ratio)
        test_set = dataset(args.dataset_path, category=args.category, train=False, normalize=args.image_normalization, preserve_aspect_ratio=args.preserve_aspect_ratio)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers,pin_memory=False, drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers,pin_memory=False, drop_last=False)

        with torch.no_grad():
            device = torch.device(args.gpu_type)
            ad_model = lwinnn(device=device,
                              backbone=args.backbone,
                              layers=('layer1', 'layer2', 'layer3'),
                              pool=args.pool,
                              interpolation_mode=args.interpolation_mode,
                              window_size=args.window_size,
                              limit_train_samples=args.limit_train_samples)
            ad_model = ad_model.to(device)

            train_start = time.time()
            ad_model.fit(train_loader)
            train_end = time.time()

            test_start = time.time()
            image_anomaly_scores, pixel_anomaly_scores = ad_model.predict(test_loader)
            test_end = time.time()

            masks = torch.stack(test_set.get_masks())
            targets = torch.Tensor(test_set.targets).int()
            
            image_metric = BinaryAUROC()
            image_metric.update(image_anomaly_scores, targets)
            image_AUROC = image_metric.compute().item()

            reshaper = transforms.Resize(masks.shape[-2:])
            pixel_metric = BinaryAUROC()
            pixel_metric.update(reshaper(pixel_anomaly_scores).flatten(), masks.int().flatten())
            pixel_AUROC = pixel_metric.compute().item()

            print(f'Anomaly detection score: {image_AUROC}. Anomaly segmentation score: {pixel_AUROC}')
            print(f'Train time: {train_end-train_start},Test time: {test_end-test_start}')

            model_parameters = vars(args)

            if args.write_scores != '':
                model_parameters['image_AUROC'] = image_AUROC.item()
                model_parameters['pixel_AUROC'] = pixel_AUROC.item()
                model_parameters['train_time'] = train_end-train_start
                model_parameters['test_time'] = test_end-test_start
                if args.limit_train_samples == -1:
                    model_parameters['train_time_persample'] = (train_end-train_start)/len(train_set)
                    model_parameters['train_samples'] = len(train_set)
                else:
                    model_parameters['train_time_persample'] = (train_end-train_start)/args.limit_train_samples
                    model_parameters['train_samples'] = args.limit_train_samples
                model_parameters['test_time_persample'] = (test_end-test_start)/len(test_set)
                    

                filename = f'results/{args.write_scores}'
                file_exists = os.path.isfile(filename)
                
                with open(filename, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=model_parameters.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(model_parameters)

    except RuntimeError as e:
        print('failed run. Error: ', e)
        model_parameters = vars(args)
        if args.write_scores != "":
            model_parameters['image_AUROC'] = -1
            model_parameters['pixel_AUROC'] = -1
            model_parameters['train_time'] = -1
            model_parameters['test_time'] = -1
            model_parameters['train_time_persample'] = -1
            model_parameters['test_time_persample'] = -1
            model_parameters['train_samples'] = 0

            filename = f'results/{args.write_scores}'
            file_exists = os.path.isfile(filename)
            
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=model_parameters.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(model_parameters)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
