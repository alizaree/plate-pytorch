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
import warnings
import ffmpeg


def read_assignment(index, vid_names, n_steps, constraints_path,TasktoSteps=None, L=None):
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
    if TasktoSteps is not None:
        steps_txt=[]
    else: 
        steps_txt=None
    with open(path, 'r') as f:
        for line in f:
            step, start, end = line.strip().split(',')
            if TasktoSteps is not None:
                step_txt=TasktoSteps[task][int(step)-1]
            start = int(math.floor(float(start)))
            end = int(math.ceil(float(end)))
            step = int(step) - 1 + base
            
            if (steps != []) and (step == steps[-1]):
                # merge two same steps
                tmp = legal_range[-1]
                legal_range[-1] = (tmp[0], end)
            else:
                steps.append(step)
                if TasktoSteps is not None:
                    steps_txt.append(step_txt)
                legal_range.append((start, end))
    
    if L is not None:
        if len(steps) < L:
            return []            

    results = extract_steps(folder_id, steps, legal_range, steps_txt=steps_txt, L=L)
        
    return results

def extract_steps(folder_id, steps, legal_range,steps_txt=None, L=None):
    N_steps = len(steps)
    results = []
    if L is not None:
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
                    'labels': steps[i:i+L],
                    'labels_txt':steps_txt[i:i+L],
                    }
                results.append(res)
            # # Wrong
            # res = {'folder_id': folder_id,
            #        'legal_range': legal_range[0:L],
            #        'labels': [0]+steps[0:L]}
            # results.append(res)


            # Correct
            #for i in range(1, N_steps-L+1):
            #    res = {'folder_id': folder_id,
            #           'legal_range': legal_range[i:i+L],
            #           'labels': steps[(i-1):i+L]}
            #    results.append(res)
            # # randomly sample 1
            # i = np.random.randint(1, N_steps-L+2)
            # res = {'folder_id': folder_id,
            #        'legal_range': legal_range[i:i+L],
            #        'labels': steps[(i-1):i+L]}
            # results.append(res)
    else:
        res = {'folder_id': folder_id,
               'legal_range': legal_range,
               'labels': steps,
               'labels_txt':steps_txt}
        results.append(res)
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
        while idx!='':
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
                 max_traj_len=None,
                 NumActionSteps=None,
                 NumFramesAroundState=2):
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
        :NumActionSteps: if not Non returns all NumActionSteps sequential steps for each video as possible trejectories
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
            self.videos_path=[os.path.join(root,'crosstask_videos', 'videos'),os.path.join(root,'crosstask_videos', 'missing_videos')]
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
            #task_vids= {'task_id': ['video_ids']}
            primary_info = read_task_info(os.path.join(
                root, 'crosstask_release', 'tasks_primary.txt'))
            test_tasks = set(primary_info['steps'].keys())
            self.TasktoTitle=primary_info['title']
            self.TasktoSteps=primary_info['steps']

            self.n_steps = primary_info['n_steps'] #{task_id: len_steps} for primary tasks
            all_tasks = set(self.n_steps.keys()) #{task_id}
            task_vids = {task: vids for task,
                         vids in task_vids.items() if task in all_tasks} #only keep primary tasks

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
                iid=0
                for idx in range(len(all_vids)):
                    task, vid = all_vids[idx]
                    video_path = os.path.join(
                        self.features_path, str(vid)+'.npy')
                    vp1=os.path.join(self.videos_path[0], str(vid)+'.mp4')
                    vp2=os.path.join(self.videos_path[1], str(vid)+'.mp4')
                    iid+=1
                    if os.path.isfile(vp1):
                        VidPath= vp1
                    elif os.path.isfile(vp2):
                        VidPath= vp2
                    else:
                        warnings.warn("Skipping missing video: "+ str(vid))
                        continue
                    json_data.append({'id': {'vid': vid, 'task': task, 'feature': video_path, 'vidPath': VidPath,
                                             'bbox': ''},'instruction_len': self.n_steps[task]})
                print('Number of missing videos',iid-len(json_data))
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
        self.M = NumFramesAroundState
        print('... Loading box annotations might take a minute ...')

        # NEW
        self.step_data = []
        for i in range(len(self.vid_names)):
            self.step_data += read_assignment(i, self.vid_names, self.n_steps, self.constraints_path,
                                              TasktoSteps=self.TasktoSteps, L=NumActionSteps)
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
                images, legal_range, M=self.M) # start state
            
            # correct
            IterLen=self.max_traj_len if self.max_traj_len is not None else len(images_start)
            frames = []
            for i in range(IterLen):
                frames.extend(images_start[min(i, len(images_start) - 1)])  # goal
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
            for i in range(IterLen):
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
    
    def ReadVideo(self,vidp):
        probe = ffmpeg.probe(vidp)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        out, _ = (
            ffmpeg
            .input(vidp)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24',r=1)
            .run(capture_stdout=True)
        )
        video = (
            np
            .frombuffer(out, np.uint8)
            .reshape([-1, height, width, 3])
        )
        return video[1:, :,:,:]
    
    def sample_single_allFrames(self, index):
        """
        Choose and Load frames per video
        :param index:
        :return:
        """
        
        step_data = self.step_data[index]
        folder_id = step_data['folder_id']
        # read frames instead of features
        frame_paths=folder_id['vidPath']
        
        images = self.ReadVideo(frame_paths)  # (T,H,W,C=3)  #sample rate=1
        legal_range = [(start_idx, end_idx) for (
            start_idx, end_idx) in step_data['legal_range'] if end_idx < images.shape[0]+1]
        # labels = step_data['labels']

        if self.args.model_type == 'woT':
            images_start, images_end = self.curate_dataset(
                images, legal_range, M=self.M) # start state
            IterLen=self.max_traj_len if self.max_traj_len is not None else len(images_start)

            frames = torch.tensor(images)
            frames_start=torch.tensor(images_start)
            frames_end=torch.tensor(images_end)

            # labels
            labels = []
            labels_data = step_data['labels']
            # Correct
            for i in range(IterLen):
                if i < len(labels_data):
                    labels.append([labels_data[i]])
                else:
                    labels.append([0])
                    
            labels_txt = []
            labels_txt_data = step_data['labels_txt']
            for i in range(IterLen):
                if i < len(labels_txt_data):
                    labels_txt.append([labels_txt_data[i]])
                else:
                    labels_txt.append([''])       
                
            # # Wrong
            # for i in range(self.args.max_traj_len+1):
            #     if i < len(labels_data):
            #         labels.append([labels_data[i]])
            #     else:
            #         labels.append([0])
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
            
            
        return frames, labels_tensor, labels_txt, frames_start, frames_end

    def __getitem__(self, index):
        frames, labels, labels_txt, frames_start,frames_end = self.sample_single_allFrames(index)
        if self.args.model_type == 'model_T':
            global_img_tensors = frames[1:2]  # torch.Size([2, 3200])    
        else:
            global_img_tensors = frames  # torch.Size([2, 3200])

        return global_img_tensors, labels, labels_txt,  frames_start,frames_end

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
