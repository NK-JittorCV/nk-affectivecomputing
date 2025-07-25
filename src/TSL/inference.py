import jittor as jt
import jittor.nn as nn
import numpy as np
import utils
import os
import json
from train import *
from eval.eval_detection import ANETdetection

def inference(net, config, logger, test_loader, test_info, step, model_file=None):
    with jt.no_grad():
        net.eval()
        if model_file is not None:
            net.load_state_dict(jt.load(model_file))
        final_res = {}
        final_res['version'] = 'VERSION 1.3'
        final_res['results'] = {}
        final_res['external_data'] = {'used': True, 'details': 'Features from I3D Network'}
        fps_dict = json.load(open(os.path.join(config.data_path,'fps_dict.json')))
        num_correct = 0.
        num_total = 0.
        load_iter = iter(test_loader)

        for i in range(len(test_loader.dataset)):
            _, _data, _label, _, _, vid_name, vid_num_seg, _ = next(load_iter)
            _data = [_data[0],_data[1]]
            vid_num_seg = vid_num_seg[0].item()
            num_segments = _data[0].shape[1]
            vid_score, cas_sigmoid_fuse, _ = net(_data)
            agnostic_score = 1 - cas_sigmoid_fuse[:,:,-1].unsqueeze(2) # B T  1-bkg
            cas_sigmoid_fuse = cas_sigmoid_fuse[:,:,:-1] # B T fg
            label_np = _label.data
            score_np = vid_score[0].data # C
            pred_np = np.zeros_like(score_np)
            pred_np[np.where(score_np < config.class_thresh)] = 0
            pred_np[np.where(score_np >= config.class_thresh)] = 1

            if pred_np.sum() == 0:
                pred_np[np.argmax(score_np)] = 1
            correct_pred = np.sum(label_np == pred_np, axis=1)


            num_correct += np.sum((correct_pred == config.num_classes).astype(np.float32))
            num_total += correct_pred.shape[0]
            cas = cas_sigmoid_fuse
            pred = np.where(score_np >= config.class_thresh)[0]
            if len(pred) == 0:
                pred = np.array([np.argmax(score_np)])
            cas_pred = cas[0].numpy()[:, pred]
            cas_pred = np.reshape(cas_pred, (num_segments, -1, 1)) # T, C, 1
            cas_pred = utils.upgrade_resolution(cas_pred, config.scale)
            proposal_dict = {}
            agnostic_score = agnostic_score.expand((-1, -1, config.num_classes))
            agnostic_score_np = agnostic_score[0].data[:, pred]
            agnostic_score_np = np.reshape(agnostic_score_np, (num_segments, -1, 1))
            agnostic_score_np = utils.upgrade_resolution(agnostic_score_np, config.scale)

            for i in range(len(config.act_thresh_cas)):
                cas_temp = cas_pred.copy()
                zero_location = np.where(cas_temp[:, :, 0] < config.act_thresh_cas[i])
                cas_temp[zero_location] = 0

                seg_list = []
                for c in range(len(pred)):
                    pos = np.where(cas_temp[:, c, 0] > 0)
                    seg_list.append(pos)

                proposals = utils.get_proposal_oic(seg_list, cas_temp, score_np, pred, config.scale, \
                                vid_num_seg, fps_dict[vid_name[0]], num_segments)

                for i in range(len(proposals)):
                    class_id = proposals[i][0][0]

                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []

                    proposal_dict[class_id] += proposals[i]
            for i in range(len(config.act_thresh_agnostic)):
                cas_temp = cas_pred.copy()

                agnostic_score_np_temp = agnostic_score_np.copy()

                zero_location = np.where(agnostic_score_np_temp[:, :, 0] < config.act_thresh_agnostic[i])
                agnostic_score_np_temp[zero_location] = 0

                seg_list = []
                for c in range(len(pred)):
                    pos = np.where(agnostic_score_np_temp[:, c, 0] > 0)
                    seg_list.append(pos)

                proposals = utils.get_proposal_oic(seg_list, cas_temp, score_np, pred, config.scale, \
                                vid_num_seg, fps_dict[vid_name[0]], num_segments)

                for i in range(len(proposals)):
                    class_id = proposals[i][0][0]

                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []
                    proposal_dict[class_id] += proposals[i]
            final_proposals = []
            for class_id in proposal_dict.keys():
                final_proposals.append(utils.nms(proposal_dict[class_id], thresh=0.5))
            final_proposals = [final_proposals[i][j] for i in range(len(final_proposals)) for j in range(len(final_proposals[i]))]
            final_res['results'][vid_name[0]] = utils.result2json(final_proposals)

        json_path = os.path.join(config.output_path, 'temp_result.json')
        with open(json_path, 'w') as f:
            json.dump(final_res, f)
            f.close()
