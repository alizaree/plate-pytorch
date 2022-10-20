import os
from os.path import join
from torchvision.transforms import Compose
import numpy as np
from PIL import Image
import torch
from data_utils import gtransforms
import json
import pickle
import cv2
import re
import math


def read_assignment(index, vid_names, n_steps, constraints_path, L=3):
    folder_id = vid_names[index]
    task = folder_id['task']
    path = os.path.join(constraints_path, folder_id['task'] + '_' + folder_id['vid'] + '.csv') # self
    
    base = 0
    for k, v in n_steps.items():
        if k == task:
            break
        base += v
    legal_range = []
    steps = []
    with open(path, 'r') as f:
        for line in f:
            step, start, end = line.strip().split(',')
            start = int(math.floor(float(start)))
            end = int(math.ceil(float(end)))
            step = int(step) - 1 + base
            
            if (steps != []) and (step == steps[-1]):
                # merge two same steps
                tmp = legal_range[-1]
                legal_range[-1] = (tmp[0], end)
            else:
                steps.append(step)
                legal_range.append((start, end))
    
    if len(steps) < L:
        return []            

    results = extract_steps(folder_id, steps, legal_range, L)
    
    return results

def extract_steps(folder_id, steps, legal_range, L=3):
    N_steps = len(steps)
    results = []
    
    if N_steps < L:
        # actually skipped
        res = {'folder_id': folder_id,
               'legal_range': legal_range,
               'labels': steps}
        results.append(res)
    else:
        # Correct
        for i in range(N_steps-L+1):
            res = {'folder_id': folder_id,
                   'legal_range': legal_range[i:i+L],
                   'labels': steps[i:i+L]}
            results.append(res)
        # # Wrong
        # res = {'folder_id': folder_id,
        #        'legal_range': legal_range[0:L],
        #        'labels': [0]+steps[0:L]}
        # results.append(res)


        # Correct
        for i in range(1, N_steps-L+1):
            res = {'folder_id': folder_id,
                   'legal_range': legal_range[i:i+L],
                   'labels': steps[(i-1):i+L]}
            results.append(res)
        # # randomly sample 1
        # i = np.random.randint(1, N_steps-L+2)
        # res = {'folder_id': folder_id,
        #        'legal_range': legal_range[i:i+L],
        #        'labels': steps[(i-1):i+L]}
        # results.append(res)
        
    return results

def get_vids(path):
    task_vids = {}
    with open(path, 'r') as f:
        for line in f:
            task, vid, url = line.strip().split(',')
            if task not in task_vids:
                task_vids[task] = []
            task_vids[task].append(vid)
    return task_vids

def read_task_info(path):
    titles = {}
    urls = {}
    n_steps = {}
    steps = {}
    with open(path, 'r') as f:
        idx = f.readline()
        while idx is not '':
            idx = idx.strip()
            titles[idx] = f.readline().strip()
            urls[idx] = f.readline().strip()
            n_steps[idx] = int(f.readline().strip())
            steps[idx] = f.readline().strip().split(',')
            next(f)
            idx = f.readline()
    return {'title': titles, 'url': urls, 'n_steps': n_steps, 'steps': steps}


class VideoFolder(torch.utils.data.Dataset):
    """
    Something-Something dataset based on *frames* extraction
    """
    def __init__(self,
                 root,
                 file_input,
                 file_labels,
                 frames_duration,
                 args=None,
                 multi_crop_test=False,
                 sample_rate=2,
                 is_test=False,
                 is_val=False,
                 num_boxes=10,
                 model=None,
                 if_augment=True,
                 max_sentence_length=None,
                 clean_inst=True,
                 max_traj_len=10):
        """
        :param root: data root path
        :param file_input: inputs path
        :param file_labels: labels path
        :param frames_duration: number of frames
        :param multi_crop_test:
        :param sample_rate: FPS
        :param is_test: is_test flag
        :param k_split: number of splits of clips from the video
        :param sample_split: how many frames sub-sample from each clip
        """
        self.in_duration = frames_duration
        self.coord_nr_frames = self.in_duration // 2
        self.is_val = is_val
        self.data_root = root
        self.args = args

        self.max_traj_len = max_traj_len
        print('self.max_traj_len: ', self.max_traj_len)
        if args.dataset == 'crosstask':
            '''
            .
            └── crosstask
                ├── crosstask_features
                └── crosstask_release
                    ├── tasks_primary.txt
                    ├── videos.csv
                    └── videos_val.csv
            '''

            val_csv_path = os.path.join(
                root, 'crosstask_release', 'videos_val.csv')
            video_csv_path = os.path.join(
                root, 'crosstask_release', 'videos.csv')
            self.features_path = os.path.join(root, 'crosstask_features')
            # baseline
            self.constraints_path = os.path.join(
                root, 'crosstask_release', 'annotations')

            all_task_vids = get_vids(video_csv_path)
            val_vids = get_vids(val_csv_path)
            if is_val:
                task_vids = val_vids
            else:
                task_vids = {task: [vid for vid in vids if task not in val_vids or vid not in val_vids[task]] for
                             task, vids in
                             all_task_vids.items()}
            primary_info = read_task_info(os.path.join(
                root, 'crosstask_release', 'tasks_primary.txt'))
            test_tasks = set(primary_info['steps'].keys())

            self.n_steps = primary_info['n_steps']
            all_tasks = set(self.n_steps.keys())
            task_vids = {task: vids for task,
                         vids in task_vids.items() if task in all_tasks}

            cross_task_data_name = 'cross_task_data_{}.json'.format(is_val)
            if os.path.exists(cross_task_data_name):
                with open(cross_task_data_name, 'r') as f:
                    self.json_data = json.load(f)
                print('Loaded {}'.format(cross_task_data_name))
            else:
                all_vids = []
                for task, vids in task_vids.items():
                    all_vids.extend([(task, vid) for vid in vids])
                json_data = []
                for idx in range(len(all_vids)):
                    task, vid = all_vids[idx]
                    video_path = os.path.join(
                        self.features_path, str(vid)+'.npy')
                    json_data.append({'id': {'vid': vid, 'task': task, 'feature': video_path, 'bbox': ''},
                                      'instruction_len': self.n_steps[task]})
                print('All primary task videos: {}'.format(len(json_data)))
                self.json_data = json_data
                with open('cross_task_data.json', 'w') as f:
                    json.dump(json_data, f)
                print('Save to {}'.format(cross_task_data_name))
        
        self.model = model
        self.num_boxes = num_boxes
        # Prepare data for the data loader
        self.prepare_data()
        # boxes_path = args.tracked_boxes
        # self.box_annotations = []
        self.M = 2
        print('... Loading box annotations might take a minute ...')

        # NEW
        self.step_data = []
        for i in range(len(self.vid_names)):
            self.step_data += read_assignment(i, self.vid_names, self.n_steps, self.constraints_path, 3)
        print('length of step data: {}'.format(len(self.step_data)))

    def prepare_data(self):
        """
        This function creates 3 lists: vid_names, labels and frame_cnts
        :return:
        """
        print("Loading label strings")
        vid_names = []
        frame_cnts = []
        for listdata in self.json_data:
            vid_names.append(listdata['id'])
            frame_cnts.append(listdata['instruction_len'])
        self.vid_names = vid_names
        self.frame_cnts = frame_cnts

    def curate_dataset(self, images, legal_range, M=2):
        images_start_list = []
        images_end_list = []
        L = len(images)
        
        for start_idx, end_idx in legal_range:
            # start state
            image_start_idx = max(0, (start_idx - M // 2))
            image_start = images[image_start_idx: image_start_idx+M]
            images_start_list.append(image_start)

            # end state
            image_end_idx = min(L, (end_idx + M // 2)) # ?
            image_end = images[image_end_idx-M: image_end_idx]
            images_end_list.append(image_end)
            
        return images_start_list, images_end_list

    def sample_single(self, index):
        """
        Choose and Load frames per video
        :param index:
        :return:
        """
        
        step_data = self.step_data[index]
        folder_id = step_data['folder_id']
        images = np.load(os.path.join(
            self.features_path, folder_id['vid']+'.npy'))[:, :1024]  # (179, 3200) 
        legal_range = [(start_idx, end_idx) for (
            start_idx, end_idx) in step_data['legal_range'] if end_idx < images.shape[0]+1]
        # labels = step_data['labels']

        if self.args.model_type == 'woT':
            images_start, images_end = self.curate_dataset(
                images, legal_range, M=self.M)

            frames = []
            # start state
            # correct
            for i in range(self.args.max_traj_len):
                frames.extend(
                    images_start[min(i, len(images_start) - 1)])  # goal
            # # wrong
            # for i in range(self.args.max_traj_len-1):
            #     frames.extend(
            #         images_start[min(i, len(images_start) - 1)])  # goal
            # end state
            frames.extend(images_end[min(i, len(images_end) - 1)])
            frames = torch.tensor(frames)

            # labels
            labels = []
            labels_data = step_data['labels']
            # Correct
            for i in range(self.args.max_traj_len):
                if i < len(labels_data):
                    labels.append([labels_data[i]])
                else:
                    labels.append([0])
            # # Wrong
            # for i in range(self.args.max_traj_len+1):
            #     if i < len(labels_data):
            #         labels.append([labels_data[i]])
            #     else:
            #         labels.append([0])
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
            
        return frames, labels_tensor

    def __getitem__(self, index):
        frames, labels = self.sample_single(
            index)
        if self.args.model_type == 'model_T':
            global_img_tensors = frames[1:2]  # torch.Size([2, 3200])    
        else:
            global_img_tensors = frames  # torch.Size([2, 3200])

        return global_img_tensors, labels

    def __len__(self):
        # return min(len(self.json_data), len(self.frame_cnts))
        return len(self.step_data)

    def load_inst_dict(self, inst_dict_path):
        print('loading cmd dict from: ', inst_dict_path)
        if inst_dict_path is None or inst_dict_path == '':
            return None
        inst_dict = pickle.load(open(inst_dict_path, 'rb'))
        inst_dict.set_max_sentence_length(self.max_sentence_length)
        return inst_dict
