import os
import sys
import torch
import random
import numpy as np
import argparse
from os import path as osp
from tqdm import tqdm
from copy import deepcopy

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from Drop_DTW.data.data_module import DataModule
from Drop_DTW.models.model_utils import cosine_sim, linear_sim
from Drop_DTW.dp.exact_dp import dtw, drop_dtw, otam, NW, lcss
from Drop_DTW.dp.dp_utils import compute_all_costs
from Drop_DTW.models.nets import EmbeddingsMapping
from Drop_DTW.models.model_utils import load_last_checkpoint
from Drop_DTW.eval.metrics import framewise_accuracy, IoU, recall_crosstask


device = "cuda" if torch.cuda.is_available() else "cpu"


def framewise_eval(dataset, model, keep_p, gamma=1, config=None):
    """ evaluate representations using framewise accuracy """
    
    accuracy = {'dp': 0, 'simple': 0}
    iou = {'dp': 0, 'simple': 0}
    for i, sample in enumerate(tqdm(dataset)):
        if sample['num_steps'] < 1:
            continue

        device = "cuda" if (model is not None and next(model.parameters()).is_cuda) else "cpu"
        if model is not None:
            frame_features = model.map_video(sample['frame_features'].to(device)).detach().cpu()
            step_features = model.map_text(sample['step_features'].to(device)).detach().cpu()
        else:
            frame_features = sample['frame_features'].cpu()
            step_features = sample['step_features'].cpu()
        sample['frame_features'] = frame_features
        sample['step_features'] = step_features
        sim = (step_features @ frame_features.T)
        if config.drop_cost == 'learn':
            distractor = model.compute_distractors(step_features.mean(0).to(device)).detach().cpu()
        else:
            distractor = None

        zx_costs, drop_costs, _ = compute_all_costs(
            sample, distractor, gamma, drop_cost_type=config.drop_cost, keep_percentile=keep_p)
        zx_costs, drop_costs = [t.detach().cpu().numpy() for t in [zx_costs, drop_costs]]
        sim = sim.detach().cpu().numpy()

        # defining matching and drop costs
        if config.dp_algo in ['DropDTW', 'NW', 'LCSS']:
            dp_fn_dict = {'DropDTW': drop_dtw, 'NW': NW, 'LCSS': lcss}
            dp_fn = dp_fn_dict[config.dp_algo]
            optimal_assignment = dp_fn(zx_costs, drop_costs, return_labels=True) - 1
        elif config.dp_algo == 'OTAM':
            _, path = otam(-sim)
            optimal_assignment = np.zeros(sim.shape[1]) - 1
            optimal_assignment[path[1]] = path[0]
        else: 
            _, path = dtw(-sim)
            _, uix = np.unique(path[1], return_index=True)
            optimal_assignment = path[0][uix]

        simple_assignment = np.argmax(sim, axis=0)
        simple_assignment[drop_costs < zx_costs.min(0)] = -1
        
        # get framewise accuracy for each vid
        accuracy['simple'] += framewise_accuracy(
            simple_assignment, sample, use_unlabeled=config.use_unlabeled)
        accuracy['dp'] += framewise_accuracy(
            optimal_assignment, sample, use_unlabeled=config.use_unlabeled)
        iou['simple'] += IoU(simple_assignment, sample)
        iou['dp'] += IoU(optimal_assignment, sample)
    num_samples = len(dataset)
    return [v / num_samples for v in
            [accuracy['simple'], accuracy['dp'], iou['simple'], iou['dp']]]

def compute_all_metrics(dataset, model, gamma, config):
    sim_matricies = []
    distractors = []
    for sample in dataset:
        if sample['num_steps'] < 1:
            continue

        device = "cuda" if (model is not None and next(model.parameters()).is_cuda) else "cpu"
        frame_features = sample['frame_features'].to(device)
        step_features = sample['step_features'].to(device)

        distractor = None
        if model is not None:
            frame_features = model.map_video(frame_features).detach().cpu()
            step_features = model.map_text(step_features).detach().cpu()
            if config.drop_cost == 'learn':
                mean_step = sample['step_features'].mean(0).to(device)
                distractor = model.compute_distractors(mean_step).detach().cpu()
        
        # get pairwise similarity
        if config.distance == 'inner':
            text_clip_similarity = linear_sim(step_features, frame_features)
        elif config.distance == 'cos':
            text_clip_similarity = cosine_sim(step_features, frame_features)
        sim_matricies.append(text_clip_similarity)
        distractors.append(distractor)

    keep_p = config.keep_percentile
    keep_p = keep_p / 3 if config.dataset == 'CrossTask' else keep_p
    keep_p = keep_p / 2 if config.dataset == 'YouCook2' else keep_p
    accuracy_std, accuracy_dtw, iou_std, iou_dtw = framewise_eval(dataset, model, keep_p, gamma, config)
    recall = recall_crosstask(dataset, model)
    return accuracy_std * 100, iou_std * 100, accuracy_dtw * 100, iou_dtw * 100, recall


if __name__ == '__main__':
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='COIN', help='dataset')
    parser.add_argument('--name', type=str, default='', help='model for evaluation, if nothing is given, evaluate pretrained features')
    parser.add_argument('--distance', type=str, default='inner', help='distance type')
    parser.add_argument('--dp_algo', type=str, default='DropDTW', choices=['DropDTW', 'OTAM', 'DTW', 'NW', 'LCSS'], help='distance type')
    parser.add_argument('--drop_cost', type=str, default='logit', help='Whather do drop in drop-dtw')
    parser.add_argument('--keep_percentile', type=float, default=0.3, help='If drop_cost is logits, the percentile to set the drop to')
    parser.add_argument('--use_unlabeled', type=bool, default=True,
                        help='use unlabeled frames in comparison (useful to consider dropped steps)')
    args = parser.parse_args()
    print(args)

    # fix random seed
    torch.manual_seed(1)
    random.seed(1)

    dataset = DataModule(args.dataset, 1, 1).val_dataset
    
    if args.name:
        gamma = 30
        model = EmbeddingsMapping(d=512, learnable_drop=(args.drop_cost == 'learn'),
                                  video_layers=2, text_layers=0)
        load_last_checkpoint(args.name, model, device, remove_name_preffix='model.')
        model.to('cuda')
        model.eval()
    else:
        model, gamma = None, 1

    accuracy_std, iou_std, accuracy_dtw, iou_dtw, recall = compute_all_metrics(
        dataset, model, gamma, args)
    
    print(f"Recall is {recall:.1f}%")
    print(f"DTW Accuracy is {accuracy_dtw:.1f}%")
    print(f"DTW IoU is {iou_dtw:.1f}%")
    
