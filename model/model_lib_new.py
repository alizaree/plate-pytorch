import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet3d_xl import Net
import numpy as np
from model import base_nets_new as base_nets
from model.model_utils import init
from queue import PriorityQueue
import operator

class BC_MODEL(nn.Module):
    def __init__(self, args):
        super(BC_MODEL, self).__init__()
        args.hidden_size = 1024
        self.base = GoalAttentionModel(args, recurrent=True, hidden_size=args.hidden_size, mode=args.gpt_repr)
        self.train()

    def forward(self, global_img_input, video_label, is_inference=False):
        value = self.base(global_img_input, video_label, is_inference=is_inference)
        return value

class GoalAttentionModel(nn.Module):
    def __init__(self, args, recurrent=False, hidden_size=128, mode='one'):
        """
        mode: one, start_goal, patch
        """
        super(GoalAttentionModel, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        self.mode = mode
        self.args = args
        self.pred_state_action = args.pred_state_action
        self.sa_type = self.args.sa_type
        self.beam_width = self.args.beam_width
        if self.mode == 'patch':
            self.hidden_size = 1024  # 256
            # patch mode
            self.patch_dim = 256
            self.num_patches = int((1024 // self.patch_dim))
            self.flatten_dim = self.patch_dim
            self.linear_encoding = nn.Linear(self.flatten_dim, self.hidden_size)
            self.max_message_len = args.max_traj_len + 4 * self.num_patches
        elif self.mode == 'start_goal':
            self.max_message_len = args.max_traj_len + 1
            self.hidden_size = hidden_size
        else:
            self.max_message_len = args.max_traj_len
            self.hidden_size = hidden_size
        if self.pred_state_action:
            if self.sa_type == 'temporal_concat':
                self.max_message_len = self.max_message_len * 2 - 1  # (se, a_0, s_0, a_1, s_1, a_2)
        self.nr_frames = 4
        self.n_steps = {'23521': 6, '59684': 5, '71781': 8, '113766': 11, '105222': 6, '94276': 6, '53193': 6,
                        '105253': 11, '44047': 8, '76400': 10, '16815': 3, '95603': 7, '109972': 5, '44789': 8,
                        '40567': 11, '77721': 5, '87706': 9, '91515': 8}
        self.task_border = np.cumsum(list(self.n_steps.values())) + 1

        self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        ## lang encoder
        print('args.num_classes: ', args.num_classes)
        self.word_embeddings = nn.Embedding(args.num_classes + 1, self.hidden_size)
        
        self.state_encode = nn.Linear(self.hidden_size, int(self.hidden_size/2))

        ## language decoder
        if self.sa_type == 'feature_concat':
            self.lang_decoding = base_nets.LangDecode(hidden_size=self.hidden_size*2, # self.hidden_size * 2,
                                                      max_message_len=self.max_message_len,
                                                      num_classes=args.num_classes,
                                                      sa_type=self.sa_type,
                                                      state_size=self.hidden_size)
        else:
            self.lang_decoding = base_nets.LangDecode(hidden_size=self.hidden_size, max_message_len=self.max_message_len,
                                                  num_classes=args.num_classes)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train()

    def forward(self, global_img_input, video_label, is_inference):
        # global_img_input: [batch_size, #step, 2, fea_dim]
        global_img_input = global_img_input.view(global_img_input.shape[0], global_img_input.shape[1] // 2, 2, global_img_input.shape[2])
        if self.pred_state_action:
            intermediate_img_input = global_img_input
        # global_img_input: [batch_size, 4, fea_dim] (start, end)
        # [0, -1] -> (start, end)
        global_img_input = global_img_input[:, [0, -1]].view(global_img_input.shape[0], -1, global_img_input.shape[-1])
        # T=4, feat=1024
        bs, T, feat = global_img_input.shape
        # self.hidden_size = 1024
        # H, W = 2
        H = W = int(np.sqrt(feat // (self.hidden_size // 4)))

        if self.mode == 'one':
            # # ==================
            # # OLD
            # # merge start and end as a whole
            # # self.nr_frames = 4 (start*2, end*2)
            # # [batch_size, 1024, 2, 2]
            # videos_features = global_img_input.view(bs, self.nr_frames * (self.hidden_size // 4), 1, H, W).float()
            # # mean feature of start/end 4 frames
            # # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            # # [batch_size, 1024, 1, 1, 1] -> [batch_size, 1024]
            # # => videos_features.view(bs, self.hidden_size, -1).mean(-1)
            # global_features = self.avgpool(videos_features).squeeze()
            # # [batch_size, 1, 1024]
            # global_features = self.dropout(global_features).unsqueeze(1)  # torch.Size([30, 1024])
            # # ==================

            # ==================
            # NEW
            # merge start and end as a whole
            # [batch_size, 1024]
            global_features = global_img_input.mean(1).float()
            # [batch_size, 1024]
            global_features = self.dropout(global_features)
            # [batch_size, 512]
            global_features = self.state_encode(global_features)
            # [batch_size, 512]
            global_features = self.dropout(global_features)
            # [batch_size, 1, 512]
            global_features = global_features.unsqueeze(1)
            # ==================

        # message: [batch_size, #step, dim]
        ######
        # NEW: append 0 at the beginning and remove the last action
        # video_label: [batch_size, #step]
        zero_labels = torch.zeros(video_label.shape[0], 1, 1).cuda()
        # video_label: [batch_size, #step, 1]
        video_label = torch.cat((zero_labels, video_label[:, :-1, :]), dim=1)
        message = self.word_embeddings(video_label.squeeze(-1).long())
        # message = self.word_embeddings(video_label.squeeze(-1).long())
        ######
        
        if self.pred_state_action:  # and self.args.dataset != 'crosstask'
            # [batch_size, #step, 2, dim]
            inter_bs, inter_T, two, feat_dim = intermediate_img_input.shape
            state_emb = []
            for i in range(inter_T):
                # # ==================
                # # OLD
                # # [batch_size, 4, 1024], two states with two frames for each    
                # inter_img_input = intermediate_img_input[:, i:(i+2)].view(global_img_input.shape[0], -1, global_img_input.shape[-1])
                # # H, W = 32 
                # # why???
                # H = W = int(np.sqrt(feat // (self.keep_size)))
                # # ==================

                # ==================
                # NEW
                # inter_img_input: [batch_size, 1, 2, 1024]
                inter_img_input = intermediate_img_input[:, i:(i+1)]
                # inter_img_input: [batch_size, 2, 1024]
                inter_img_input = inter_img_input.view(bs, -1, feat)

                # ==================
                if self.mode in ['one', 'patch']:
                    # # ==================
                    # # OLD
                    # # self.nr_frames = 4
                    # # self.keep_size = 1
                    # # [batch_size, 4, 1, 32, 32]
                    # # ?????????? 1024d -> 1d feature???
                    # inter_videos_features = inter_img_input.view(bs, self.nr_frames * self.keep_size, 1, H, W).float()
                    # # [batch_size, 4] 
                    # inter_global_features = self.avgpool(inter_videos_features).squeeze()
                    # # [batch_size, 1, 4]
                    # inter_global_features = self.dropout(inter_global_features).unsqueeze(1)  # torch.Size([30, 1024])
                    # # ==================

                    # ==================
                    # NEW
                    # [batch_size, 1024]
                    inter_videos_features = inter_img_input.mean(1).float()
                    # [batch_size, 1024]
                    inter_videos_features = self.dropout(inter_videos_features)
                    # [batch_size, 512]
                    inter_videos_features = self.state_encode(inter_videos_features)
                    # [batch_size, 512]
                    inter_videos_features = self.dropout(inter_videos_features)
                    # [batch_size, 1, 512]
                    inter_videos_features = inter_videos_features.unsqueeze(1)
                    # ==================
                state_emb.append(inter_videos_features)
            # [batch_size, #step, 1024/2]
            # obtain message to predict states
            # message: state + action
            state_emb = torch.cat(state_emb, dim=1)
            state_emb_target = state_emb[:, 1:, :].clone()
            state_emb = state_emb[:, :-1, :]
            if self.sa_type == 'feature_concat':
                # global_features: [batch_size, #step, 1024/2]
                global_features = global_features.repeat(1, T-1, 1).float()
                # [batch_size, #step, 1024+1024]
                # intermediate + action
                message = torch.cat((global_features, state_emb, message), dim=-1)
                
        if self.pred_state_action and self.sa_type == 'feature_concat':
            message_input = message  # torch.cat((state_emb, message), dim=-1)
        else:
            message_input = torch.cat([global_features, message], dim=1)
        if self.pred_state_action:
            if self.sa_type == 'feature_concat':
                cls_output, state_feat = self.lang_decoding.get_feat(message_input)
                cls_output = cls_output #[:, :-1]
                state_feat = state_feat #[:, :-2]
        else:
            cls_output = self.lang_decoding(message_input)[:, :-1]
        if self.mode == 'start_goal':
            cls_output = cls_output[:, 1:]
            if self.pred_state_action:
                state_feat = state_feat[:, 1:]
        elif self.mode == 'patch':
            cls_output = cls_output[:, (T * self.num_patches - 1):]
            if self.pred_state_action:
                state_feat = state_feat[:, (T * self.num_patches - 1):]
        if self.pred_state_action:
            if self.sa_type == 'temporal_concat':
                state_feat = state_feat[:, 1::2]
                state_loss = torch.tensor(0.)
                cls_output = cls_output[:, 0::2]
            elif self.sa_type == 'feature_concat':
                # state_loss = 0.1 * F.mse_loss(state_feat, state_emb[:, :-1], reduction='mean') * 0.1  # / state_emb.shape[0] * 0.1
                state_loss = F.mse_loss(state_feat, state_emb, reduction='mean') # / state_emb.shape[0] * 0.1
        else:
            state_loss = torch.tensor(0.)
        return cls_output, state_loss

    def model_get_action(self, global_img_input, video_label, is_inference):
        if self.args.dataset == 'crosstask':
            global_img_input = global_img_input.view(global_img_input.shape[0], global_img_input.shape[1] // 2, 2,
                                                     global_img_input.shape[2])
            inter_img_input = global_img_input[:, :1, :, :]
            global_img_input = global_img_input[:, [0, -1]].view(global_img_input.shape[0], -1, global_img_input.shape[-1])
            bs, T, feat = global_img_input.shape  # [30, 2, 3200]  # torch.Size([30, 4, 1024])
           
        if self.mode == 'one':
            # merge start and end as a whole
            # [batch_size, 1024]
            global_features = global_img_input.mean(1).float()
            # [batch_size, 1024]
            # global_features = self.dropout(global_features).unsqueeze(1)
            global_features = self.state_encode(global_features)
            # [batch_size, 1, 512]
            global_features = global_features.unsqueeze(1).float()
            

        if self.pred_state_action and self.sa_type == 'feature_concat':
            # [batch_size, 1, 1024]
            inter_img_input = inter_img_input.mean(2).float()
            # [batch_size, 1, 512]
            inter_img_input = self.state_encode(inter_img_input)
            # video_label: [batch_size, 1]
            zero_labels = torch.zeros(video_label.shape[0], 1).cuda()
            # message: [batch_size, 1, 1024]
            message = self.word_embeddings(zero_labels.long())
            global_features = torch.cat((global_features, inter_img_input, message), dim=-1)

        max_message_len = self.args.max_traj_len
        domain_prior_list = []
        if self.args.search_method == 'beam':
            message, domain_prior_list = self.beam_decode(max_message_len, global_features, task_border=self.task_border)
        else:
            message = self.gen_message(max_message_len, global_features, message)
        return message, domain_prior_list

    # def gen_message(self, max_message_len, belief_goal_context, message, sample=True):
    #     temperature = 0.9
    #     sampled_ids = []
    #     sampled_probs = []
    #     message_previous = belief_goal_context
    #     for message_step in range(max_message_len):
    #         if self.sa_type == 'feature_concat':
    #             message_output, state_feat = self.lang_decoding.get_feat(message_previous)
    #             state_feat = state_feat[:, -1:, :]
    #         else:
    #             message_output = self.lang_decoding(message_previous)
    #         message_output = message_output[:, -1, :] / temperature
    #         message_probs = F.softmax(message_output, dim=-1)
    #         if sample:
    #             message_prediction = torch.multinomial(message_probs, num_samples=1)
    #         else:
    #             _, message_prediction = torch.topk(message_probs, k=1, dim=-1)
    #         message = self.word_embeddings(message_prediction)  # [1, 1, 64]
    #         if self.sa_type == 'feature_concat':
    #             message = torch.cat([state_feat, message], dim=-1)
    #         message_previous = torch.cat([message_previous, message], dim=1)
    #         sampled_ids.append(message_prediction[:, -1])
    #         sampled_probs.append(message_probs.unsqueeze(1))
    #         if self.pred_state_action and message_step < (max_message_len-1) and self.sa_type == 'temporal_feature':
    #             message_output = self.lang_decoding(message_previous)
    #             message_output = message_output[:, -1, :] / temperature
    #             message_probs = F.softmax(message_output, dim=-1)
    #             if sample:
    #                 message_prediction = torch.multinomial(message_probs, num_samples=1)
    #             else:
    #                 _, message_prediction = torch.topk(message_probs, k=1, dim=-1)
    #             message = self.word_embeddings(message_prediction)  # [1, 1, 64]
    #             message_previous = torch.cat([message_previous, message], dim=1)

    #     message_next = torch.stack(sampled_probs, dim=1)
    #     return message_next

    def beam_decode(self, max_message_len, belief_goal_context, task_border, use_task_border=False):
        '''
        # https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''

        topk = 1  # how many sentence do you want to generate
        decoded_batch = []
        temperature = 0.9
        domain_prior_list = []


        # decoding goes sentence by sentence
        for idx in range(belief_goal_context.size(0)):
            # Start with the start of the sentence token
            message_previous = belief_goal_context[idx:(idx+1)]
            global_features = message_previous[:, :, :512]

            # Number of sentence to generate
            endnodes = []

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(hiddenstate=message_previous, previousNode=None, wordId="", logProb=0, length=1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            domain_prior_list_each_img = []
            # start beam search
            for message_step in range(max_message_len):
                # fetch the best node
                prev_score, n = nodes.get()
                # decoder_input = n.wordid
                message_previous = n.h
                if self.sa_type == 'feature_concat':
                    message_output, state_feat = self.lang_decoding.get_feat(message_previous)
                    state_feat = state_feat[:, -1:, :]
                else:
                    message_output = self.lang_decoding(message_previous)
                message_output = message_output[:, -1, :] / temperature
                message_probs = F.softmax(message_output, dim=-1)

                # PUT HERE REAL BEAM SEARCH OF TOP
                if self.args.dataset == 'crosstask' and use_task_border:
                    _, indexes_1 = torch.topk(message_probs, self.beam_width)
                    valid_interval_start = valid_interval_end = None
                    for task_idx in range(task_border.shape[0]):
                        if task_idx == 0 and (indexes_1[0][0] < task_border[task_idx]):
                            valid_interval_start = 0
                            valid_interval_end = task_border[task_idx]
                        elif task_idx > 0 and indexes_1[0][0] >= task_border[task_idx-1] \
                                and indexes_1[0][0] < task_border[task_idx]:
                            valid_interval_start = task_border[task_idx-1]
                            valid_interval_end = task_border[task_idx]
                        elif indexes_1[0][0] == 133:
                            valid_interval_start = task_border[-2]
                            valid_interval_end = task_border[-1]
                            assert valid_interval_end == 133, 'valid_interval_end: {}'.format(valid_interval_end)
                    assert valid_interval_start is not None and valid_interval_end is not None, \
                        'valid_interval_start: {}, valid_interval_end: {}, indexes_1: {}'.format(valid_interval_start, valid_interval_end, indexes_1)

                    message_probs[:, :valid_interval_start] = 0.
                    message_probs[:, valid_interval_end:] = 0.

                log_prob, indexes = torch.topk(message_probs, self.beam_width)
                if use_task_border:
                    if self.args.dataset == 'crosstask':
                        domain_prior_list_each_img.append(1. - torch.prod((indexes == indexes_1).float()).cpu().numpy())
                    else:
                        domain_prior_list_each_img.append(0)

                nextnodes = []

                for new_k in range(self.beam_width):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    message = self.word_embeddings(decoded_t)  # [1, 1, 64]
                    if self.sa_type == 'feature_concat':
                        message = torch.cat([global_features, state_feat, message], dim=-1)
                    message_previous = torch.cat([message_previous, message], dim=1)  # torch.Size([1, 2, 1024])
                    if self.pred_state_action and message_step < (max_message_len-1) and self.sa_type == 'temporal_feature':
                        message_output = self.lang_decoding(message_previous)
                        message_output = message_output[:, -1, :] / temperature
                        message_probs = F.softmax(message_output, dim=-1)
                        _, message_prediction = torch.topk(message_probs, k=1, dim=-1)
                        message = self.word_embeddings(message_prediction)  # [1, 1, 64]
                        message_previous = torch.cat([message_previous, message], dim=1)

                    log_p = log_prob[0][new_k].item()  # the larger, the better
                    node = BeamSearchNode(hiddenstate=message_previous, previousNode=n, wordId=decoded_t,
                                          logProb=n.logp + log_p, length=n.leng + 1, message_prob=log_p)
                    score = -node.eval()  # the less, the better
                    nextnodes.append((score, node))
                    assert score < prev_score, 'score: {} should < prev_score: {}'.format(score, prev_score)

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
            if use_task_border:
                domain_prior_list.append(max(domain_prior_list_each_img))
            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                # The lowest valued entries are retrieved first, -node.eval()
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                while n.prevNode != None:
                    n = n.prevNode
                    if n.wordid != "":
                        utterance.append(n.wordid)
                utterance = utterance[::-1]
                utterance = torch.stack(utterance, dim=1)
                utterances.append(utterance)
                utterances = torch.cat(utterances, dim=0)

            decoded_batch.append(utterances)
        decoded_batch = torch.cat(decoded_batch, dim=0)
        decoded_batch = torch.nn.functional.one_hot(decoded_batch, num_classes=self.args.num_classes).unsqueeze(-2).double()
        return decoded_batch, domain_prior_list

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length, message_prob=None):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length
        self.message_prob = message_prob

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.logp + alpha * reward

    # defining comparators less_than and equals
    def __lt__(self, other):
        return self.logp < other.logp
