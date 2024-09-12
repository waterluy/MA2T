# ---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
# ---------------------------------------------------------------------------------#

import torch
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
import copy
import os
from ..dense_heads.seg_head_plugin import IOU
from .uniad_track import UniADTrack
from mmdet.models.builder import build_head
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
import numpy as np


@DETECTORS.register_module()
class UniAD(UniADTrack):
    """
    UniAD: Unifying Detection, Tracking, Segmentation, Motion Forecasting, Occupancy Prediction and Planning for Autonomous Driving
    """

    def __init__(
        self,
        seg_head=None,
        motion_head=None,
        occ_head=None,
        planning_head=None,
        task_loss_weight=dict(track=1.0, map=1.0, motion=1.0, occ=1.0, planning=1.0),
        **kwargs,
    ):
        super(UniAD, self).__init__(**kwargs)
        if seg_head:
            self.seg_head = build_head(seg_head)
        if occ_head:
            self.occ_head = build_head(occ_head)
        if motion_head:
            self.motion_head = build_head(motion_head)
        if planning_head:
            self.planning_head = build_head(planning_head)

        self.task_loss_weight = task_loss_weight
        assert set(task_loss_weight.keys()) == {
            "track",
            "occ",
            "motion",
            "map",
            "planning",
        }
        # self.adv_train = False    在BaseDetector里
        # self.pgd_cfg = PGDCfg()   在UniADTrack里
        # self.attack = False       在UniADTrack里
        self.attack_mode = "all"
        self.gen_adv_img = False
        self.index = 0
        self.yinta = 0.0002
        self.noise_img = None
        self.agnostic = False
        self.noise_weights = None
        # self.same_weights = False
        self.freeze_perception = False
        self.freeze_prediction = False
        self.freeze_plan = False
        self.attack_method = "pgdoo"
        self.n_tasks = 5
        self.ma2t_task_weights = np.ones(self.n_tasks, dtype=np.float32)
        self.task_dict = {"track": 0, "map": 1, "motion": 2, "occ": 3, "plan": 4}
        self.save_noise = None
        self.black = False
        self.transfer = None

    @property
    def with_planning_head(self):
        return hasattr(self, "planning_head") and self.planning_head is not None

    @property
    def with_occ_head(self):
        return hasattr(self, "occ_head") and self.occ_head is not None

    @property
    def with_motion_head(self):
        return hasattr(self, "motion_head") and self.motion_head is not None

    @property
    def with_seg_head(self):
        return hasattr(self, "seg_head") and self.seg_head is not None

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def set_dynamic(self, dynamic):
        self.dynamic = dynamic

    def freeze(self, stage=None):
        if stage == "perception":
            self.freeze_perception = True
            print("freeze perception")
            self.pts_bbox_head.freeze_params()
            self.seg_head.freeze_params()
        elif stage == "bbox":
            print("freeze bbox")
            self.pts_bbox_head.freeze_params()
        elif stage == "prediction":
            self.freeze_prediction = True
            print("freeze prediction")
            self.motion_head.freeze_params()
            self.occ_head.freeze_params()
        elif stage == "plan":
            self.freeze_plan = True
            print("freeze plan")
            self.planning_head.freeze_params()
        elif stage == "occ":
            print("freeze occ")
            self.occ_head.freeze_params()
        elif stage == "motion":
            print("freeze motion")
            self.motion_head.freeze_params()
        elif stage == "map":
            print("freeze map")
            self.seg_head.freeze_params()
        else:
            print("freeze none")
            quit()

    def init_noise_weights(self):
        # device=next(self.parameters()).device
        self.noise_weights = torch.tensor(
            [1, 1, 1, 1, 1],
            dtype=torch.float,
        ).cuda()
        self.noise_weights.requires_grad_(True)
        # print(self.noise_weights)
        # quit()

    def get_task_loss(self, task_losses, task_name, weight=None):
        for k, v in task_losses.items():
            task_losses[k] = torch.nan_to_num(v)
        from collections import OrderedDict

        log_vars = OrderedDict()
        for loss_name, loss_value in task_losses.items():
            if task_name not in loss_name:
                continue
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)
        if weight != None:
            return loss * weight
        else:
            return loss

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            if self.adv_train:                
                    if (self.dynamic is not None) and ("ma2t" in self.dynamic):
                        self.at_attack = True
                        self.at_train_model = False
                        return self.forward_adv_train_ma2t(ori_loss=None, **kwargs)
                    else:
                        self.at_attack = True
                        self.at_train_model = False
                        return self.forward_adv_train(ori_loss=None, **kwargs)
            else:
                return self.forward_train(**kwargs)
        else:
            if self.attack:
                if self.attack_mode == "task":
                    return self.forward_attack_v1_task_loss_res(**kwargs)
                elif self.attack_mode == "plan":
                    return self.forward_attack_plan_loss_res(**kwargs)
                elif self.attack_mode == "all":
                    return self.forward_attack_all_loss_res_imgok(**kwargs)
                else:
                    print("error")
                    quit()
            else:
                return self.forward_test(**kwargs)

    def forward_adv_train(
        self,
        ori_loss=None,
        img=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_inds=None,
        l2g_t=None,
        l2g_r_mat=None,
        timestamp=None,
        gt_lane_labels=None,
        gt_lane_bboxes=None,
        gt_lane_masks=None,
        rescale=False,
        gt_fut_traj=None,
        gt_fut_traj_mask=None,
        gt_past_traj=None,
        gt_past_traj_mask=None,
        gt_sdc_bbox=None,
        gt_sdc_label=None,
        gt_sdc_fut_traj=None,
        gt_sdc_fut_traj_mask=None,
        # Occ_gt
        gt_segmentation=None,
        gt_instance=None,
        gt_occ_img_is_valid=None,
        # planning
        sdc_planning=None,
        sdc_planning_mask=None,
        command=None,
        # fut gt for planning
        gt_future_boxes=None,
        **kwargs,  # [1, 9]
    ):
        if self.pgd_cfg.attack_img and self.pgd_cfg.random_start:
            noise_img = torch.empty_like(img).uniform_(
                -self.pgd_cfg.img_eps, self.pgd_cfg.img_eps
            )
            noise_img.requires_grad_(True)
        else:
            noise_img = torch.zeros_like(img)
        if self.pgd_cfg.attack_img:
            noise_img.requires_grad_(True)

        mi_noise_img_grad = 0
        mi_noise_track_grad = 0
        mi_noise_seg_grad = 0
        mi_noise_motion_traj_grad = 0
        mi_noise_motion_sdc_traj_grad = 0

        if (
            self.pgd_cfg.attack_img
            or self.pgd_cfg.attack_track_motion
            or self.pgd_cfg.attack_seg_motion
            or self.pgd_cfg.attack_motion_occ
            or self.pgd_cfg.attack_motion_plan
        ):
            lun = self.pgd_cfg.steps + 1
        else:
            lun = 1

        for pgd_i in range(lun):

            # if pgd_i == lun - 1:
            #     if not self.freeze_perception:
            #         for name, param in self.pts_bbox_head.named_parameters():
            #             if name != 'code_weights':
            #                 param.requires_grad = True
            #     self.at_attack = False
            #     self.at_train_model = True
            # print(pgd_i)
            adv_img = img
            if self.pgd_cfg.attack_img:
                adv_img = img + noise_img

            losses = dict()
            # print(type(img))    # <class 'torch.Tensor'>
            len_queue = adv_img.size(1)  # queue_length
            # print(img.shape)    # torch.Size([1, 3, 6, 3, 928, 1600])
            # quit()
            track_img_metas = copy.deepcopy(img_metas)  # !!!!!!!!!!!!
            losses_track, outs_track = self.forward_track_train(
                adv_img,
                gt_bboxes_3d,
                gt_labels_3d,
                gt_past_traj,
                gt_past_traj_mask,
                gt_inds,
                gt_sdc_bbox,
                gt_sdc_label,
                l2g_t,
                l2g_r_mat,
                track_img_metas,
                timestamp,
            )
            losses_track = self.loss_weighted_and_prefixed(losses_track, prefix="track")
            losses.update(losses_track)
            # print(losses_track[list(losses_track.keys())[0]])

            # Upsample bev for tiny version
            outs_track = self.upsample_bev_if_tiny(
                outs_track
            ) 
            bev_embed = outs_track["bev_embed"]
            bev_pos = outs_track["bev_pos"]

            # outs_track.keys()
            # dict_keys(['bev_embed', 'bev_pos', 'track_query_embeddings', 'track_query_matched_idxes', 'track_bbox_results', 'sdc_boxes_3d', 'sdc_scores_3d', 'sdc_track_scores', 'sdc_track_bbox_results', 'sdc_embedding'])
            # print(outs_track['track_query_embeddings'].shape)    # [37, 256]  [39, 256]  大小不固定
            # print(outs_track['sdc_embedding'].shape)
            if 0 == pgd_i:
                assert (
                    outs_track["track_query_embeddings"].shape[-1]
                    == outs_track["sdc_embedding"].shape[-1]
                )
                cat_outs_track = torch.cat(
                    [
                        outs_track["sdc_embedding"][None, :],
                        outs_track["sdc_embedding"][None, :],
                    ],
                    dim=0,
                )
                # print(cat_outs.shape) [2, 256]
                if self.pgd_cfg.attack_track_motion:
                    if self.pgd_cfg.random_start:
                        noise_track = torch.empty_like(cat_outs_track).uniform_(
                            -self.pgd_cfg.track_eps, self.pgd_cfg.track_eps
                        )
                    else:
                        noise_track = torch.zeros_like(cat_outs_track)
                    noise_track.requires_grad_(True)
                else:
                    noise_track = torch.zeros_like(cat_outs_track)
            ############ 分别加noise ############
            adv_outs_track = outs_track
            if self.pgd_cfg.attack_track_motion:
                # print(noise_track[0:-1, :].shape)
                # print(noise_track[-1, :].shape)
                if adv_outs_track["track_query_embeddings"].shape[0] != 0:
                    adv_outs_track["track_query_embeddings"] = adv_outs_track[
                        "track_query_embeddings"
                    ] + torch.cat(
                        [
                            noise_track[0, :][None, :]
                            for _ in range(
                                adv_outs_track["track_query_embeddings"].shape[0]
                            )
                        ],
                        dim=0,
                    )  # noise_track[0:-1, :]
                adv_outs_track["sdc_embedding"] = (
                    adv_outs_track["sdc_embedding"] + noise_track[-1, :]
                )

            seg_img_metas = copy.deepcopy(track_img_metas)  # !!!!!!!!!!!!
            seg_img_metas = [each[len_queue - 1] for each in seg_img_metas]

            outs_seg = dict()
            if self.with_seg_head:
                losses_seg, outs_seg = self.seg_head.forward_train(
                    bev_embed,
                    seg_img_metas,
                    gt_lane_labels,
                    gt_lane_bboxes,
                    gt_lane_masks,
                )
                losses_seg = self.loss_weighted_and_prefixed(losses_seg, prefix="map")
                losses.update(losses_seg)

            if 0 == pgd_i:
                assert (
                    outs_seg["args_tuple"][3].shape == outs_seg["args_tuple"][5].shape
                )
                cat_outs_seg = torch.cat(
                    [outs_seg["args_tuple"][3][None], outs_seg["args_tuple"][5][None]],
                    dim=0,
                )
                if self.pgd_cfg.attack_seg_motion:
                    if self.pgd_cfg.random_start:
                        noise_seg = torch.empty_like(cat_outs_seg).uniform_(
                            -self.pgd_cfg.seg_eps, self.pgd_cfg.seg_eps
                        )
                    else:
                        noise_seg = torch.zeros_like(cat_outs_seg)
                    noise_seg.requires_grad_(True)
                else:
                    noise_seg = torch.zeros_like(cat_outs_seg)
            adv_outs_seg = outs_seg
            if self.pgd_cfg.attack_seg_motion:
                adv_outs_seg["args_tuple"][3] = (
                    adv_outs_seg["args_tuple"][3] + noise_seg[0]
                )
                adv_outs_seg["args_tuple"][5] = (
                    adv_outs_seg["args_tuple"][5] + noise_seg[1]
                )

            outs_motion = dict()
            # Forward Motion Head
            if self.with_motion_head:
                ret_dict_motion = self.motion_head.forward_train(
                    bev_embed,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    gt_fut_traj,
                    gt_fut_traj_mask,
                    gt_sdc_fut_traj,
                    gt_sdc_fut_traj_mask,
                    outs_track=adv_outs_track,
                    outs_seg=adv_outs_seg,
                )
                losses_motion = ret_dict_motion["losses"]
                outs_motion = ret_dict_motion["outs_motion"]
                outs_motion["bev_pos"] = bev_pos
                losses_motion = self.loss_weighted_and_prefixed(
                    losses_motion, prefix="motion"
                )
                losses.update(losses_motion)

            if self.with_occ_head:
                # assert 0 != outs_motion['track_query'].shape[1]
                if outs_motion["track_query"].shape[1] == 0:  # 是可能出现的
                    # TODO: rm hard code
                    outs_motion["track_query"] = torch.zeros((1, 1, 256)).to(bev_embed)
                    outs_motion["track_query_pos"] = torch.zeros((1, 1, 256)).to(
                        bev_embed
                    )
                    outs_motion["traj_query"] = torch.zeros((3, 1, 1, 6, 256)).to(
                        bev_embed
                    )
                    outs_motion["all_matched_idxes"] = [[-1]]
            # print(outs_motion.get('traj_query', None).shape)    # torch.Size([3, 1, 36, 6, 256])
            # print(outs_motion['track_query'].shape)             # torch.Size([1, 36, 256])
            # print(outs_motion['track_query_pos'].shape)         # torch.Size([1, 36, 256])  36不固定
            if 0 == pgd_i:
                assert "traj_query" in outs_motion.keys()
                assert (
                    outs_motion["track_query"].shape
                    == outs_motion["track_query_pos"].shape
                )
                assert (
                    outs_motion["track_query"].shape[0]
                    == outs_motion["traj_query"].shape[1]
                )
                assert (
                    outs_motion["track_query"].shape[1]
                    == outs_motion["traj_query"].shape[2]
                )
                assert (
                    outs_motion["track_query"].shape[-1]
                    == outs_motion["traj_query"].shape[-1]
                )
                tmp_traj_query = outs_motion["traj_query"][-1]  # 只有 [-1] 被用到了
                tmp_traj_query = tmp_traj_query[:, 0:1, :, :]  # 必须用 0:1 不用能用0
                # tmp_track_query = outs_motion['track_query'][None, :, 0:1, :]
                # track_query_pos detach了  !!!!!!
                # tmp_track_query_pos = outs_motion['track_query_pos'][None, :, 0:1, :]
                # cat_outs_motion_track = torch.cat([tmp_track_query, tmp_track_query_pos], dim=0)
                if self.pgd_cfg.attack_motion_occ:
                    if self.pgd_cfg.random_start:
                        noise_motion_traj = torch.empty_like(tmp_traj_query).uniform_(
                            -self.pgd_cfg.motion_traj_eps, self.pgd_cfg.motion_traj_eps
                        )
                        # noise_motion_track = torch.empty_like(tmp_track_query).uniform_(-self.pgd_cfg.motion_track_eps, self.pgd_cfg.motion_track_eps)
                    else:
                        noise_motion_traj = torch.zeros_like(tmp_traj_query)
                    noise_motion_traj.requires_grad_(True)
                else:
                    noise_motion_traj = torch.zeros_like(tmp_traj_query)
            adv_outs_motion = outs_motion
            if self.pgd_cfg.attack_motion_occ:
                assert "traj_query" in adv_outs_motion.keys()
                # 1, 36, 6, 256
                if outs_motion["track_query"].shape[1] != 0:
                    adv_outs_motion["traj_query"][-1] = adv_outs_motion["traj_query"][
                        -1
                    ] + torch.cat(
                        [
                            noise_motion_traj
                            for _ in range(adv_outs_motion["traj_query"][-1].shape[1])
                        ],
                        dim=1,
                    )  

            # Forward Occ Head
            if self.with_occ_head:
                losses_occ = self.occ_head.forward_train(
                    bev_embed,
                    adv_outs_motion,
                    gt_inds_list=gt_inds,
                    gt_segmentation=gt_segmentation,
                    gt_instance=gt_instance,
                    gt_img_is_valid=gt_occ_img_is_valid,
                )
                losses_occ = self.loss_weighted_and_prefixed(losses_occ, prefix="occ")
                losses.update(losses_occ)

            # outs_motion['sdc_traj_query'].shape    torch.Size([3, 1, 6, 256])
            # sdc_track_query detach
            if 0 == pgd_i:
                # 只用了-1
                tmp_sdc_traj_query = adv_outs_motion["sdc_traj_query"][-1]  # 1,6,256
                if self.pgd_cfg.attack_motion_plan:
                    if self.pgd_cfg.random_start:
                        noise_motion_sdc_traj = torch.empty_like(
                            tmp_sdc_traj_query
                        ).uniform_(
                            -self.pgd_cfg.motion_traj_eps, self.pgd_cfg.motion_traj_eps
                        )
                    else:
                        noise_motion_sdc_traj = torch.zeros_like(tmp_sdc_traj_query)
                    noise_motion_sdc_traj.requires_grad_(True)
                else:
                    noise_motion_sdc_traj = torch.zeros_like(tmp_sdc_traj_query)
            # print(noise_motion_sdc_traj.requires_grad)
            if self.pgd_cfg.attack_motion_plan:
                tmp = [
                    torch.zeros_like(adv_outs_motion["sdc_traj_query"][-1])
                    for _ in range(adv_outs_motion["sdc_traj_query"].shape[0] - 1)
                ]
                # add = torch.stack(tmp + [noise_motion_sdc_traj], dim=0)
                # print(add.shape)
                # print(adv_outs_motion['sdc_traj_query'].shape)
                adv_outs_motion["sdc_traj_query"] = adv_outs_motion[
                    "sdc_traj_query"
                ] + torch.stack(tmp + [noise_motion_sdc_traj], dim=0)

            # adv_outs_motion['sdc_traj_query'] = adv_outs_motion['sdc_traj_query'] + torch.ones_like(adv_outs_motion['sdc_traj_query'])
            # adv_outs_motion['sdc_traj_query'][-1] = adv_outs_motion['sdc_traj_query'][-1] + torch.ones_like(adv_outs_motion['sdc_traj_query'][-1])
            # Forward Plan Head
            if self.with_planning_head:
                outs_planning = self.planning_head.forward_train(
                    bev_embed,
                    adv_outs_motion,
                    sdc_planning,
                    sdc_planning_mask,
                    command,
                    gt_future_boxes,
                )
                losses_planning = outs_planning["losses"]
                losses_planning = self.loss_weighted_and_prefixed(
                    losses_planning, prefix="planning"
                )
                losses.update(losses_planning)

            for k, v in losses.items():
                losses[k] = torch.nan_to_num(v)

            adv_loss, adv_log_vars = self._parse_losses(losses)
            # mmcv.runner.OptimizerHook
            if pgd_i < self.pgd_cfg.steps:
                if (
                    self.pgd_cfg.attack_img
                    or self.pgd_cfg.attack_track_motion
                    or self.pgd_cfg.attack_seg_motion
                    or self.pgd_cfg.attack_motion_occ
                    or self.pgd_cfg.attack_motion_plan
                ):
                    noises = [
                        noise_img,
                        noise_track,
                        noise_seg,
                        noise_motion_traj,
                        noise_motion_sdc_traj,
                    ]
                    noises_require_grad = list(
                        filter(lambda n: n.requires_grad, noises)
                    )
                    # print(len(noises_require_grad))
                    grads = torch.autograd.grad(
                        outputs=adv_loss,
                        inputs=noises_require_grad,
                        allow_unused=True, 
                    )  
                    # print(noise_img_grad == None)
                index = 0
                if self.pgd_cfg.attack_img:
                    noise_img_grad = grads[index]
                    index = index + 1
                    assert noise_img_grad is not None
                    assert (
                        not noise_img_grad.detach()
                        .cpu()
                        .equal(torch.zeros(noise_img_grad.shape))
                    )
                    mi_noise_img_grad = (
                        self.pgd_cfg.miu * mi_noise_img_grad + noise_img_grad
                    )  #  / torch.norm(noise_img_grad, p=2, dim=None, keepdim=False, out=None, dtype=None)
                    noise_img = (
                        noise_img + self.pgd_cfg.img_alpha * mi_noise_img_grad.sign()
                    )
                    noise_img = noise_img.detach()
                    noise_img.requires_grad_(True)
                if self.pgd_cfg.attack_track_motion:
                    noise_track_grad = grads[index]
                    index = index + 1
                    assert noise_track_grad is not None
                    assert (
                        not noise_track_grad.detach()
                        .cpu()
                        .equal(torch.zeros(noise_track_grad.shape))
                    )
                    mi_noise_track_grad = (
                        self.pgd_cfg.miu * mi_noise_track_grad + (1 - self.pgd_cfg.miu) * noise_track_grad
                    )
                    noise_track = (
                        noise_track
                        + self.pgd_cfg.track_alpha * mi_noise_track_grad.sign()
                    )
                    noise_track = noise_track.detach()
                    noise_track.requires_grad_(True)
                if self.pgd_cfg.attack_seg_motion:
                    noise_seg_grad = grads[index]
                    index = index + 1
                    assert noise_seg_grad is not None
                    assert (
                        not noise_seg_grad.detach()
                        .cpu()
                        .equal(torch.zeros(noise_seg_grad.shape))
                    )
                    mi_noise_seg_grad = (
                        self.pgd_cfg.miu * mi_noise_seg_grad + (1 - self.pgd_cfg.miu) * noise_seg_grad
                    )
                    noise_seg = (
                        noise_seg + self.pgd_cfg.seg_alpha * mi_noise_seg_grad.sign()
                    )
                    noise_seg = noise_seg.detach()
                    noise_seg.requires_grad_(True)
                if self.pgd_cfg.attack_motion_occ:
                    noise_motion_traj_grad = grads[index]
                    index = index + 1
                    assert noise_motion_traj_grad is not None
                    mi_noise_motion_traj_grad = (
                        self.pgd_cfg.miu * mi_noise_motion_traj_grad
                        + (1 - self.pgd_cfg.miu) * noise_motion_traj_grad
                    )
                    noise_motion_traj = (
                        noise_motion_traj
                        + self.pgd_cfg.motion_traj_alpha
                        * mi_noise_motion_traj_grad.sign()
                    )
                    noise_motion_traj = noise_motion_traj.detach()
                    noise_motion_traj.requires_grad_(True)
                if self.pgd_cfg.attack_motion_plan:
                    noise_motion_sdc_traj_grad = grads[index]
                    index = index + 1
                    assert noise_motion_sdc_traj_grad is not None
                    mi_noise_motion_sdc_traj_grad = (
                        self.pgd_cfg.miu * mi_noise_motion_sdc_traj_grad
                        + (1 - self.pgd_cfg.miu) * noise_motion_sdc_traj_grad
                    )
                    noise_motion_sdc_traj = (
                        noise_motion_sdc_traj
                        + self.pgd_cfg.motion_traj_alpha
                        * mi_noise_motion_sdc_traj_grad.sign()
                    )
                    noise_motion_sdc_traj = noise_motion_sdc_traj.detach()
                    noise_motion_sdc_traj.requires_grad_(True)

        if ori_loss is not None:
            train_loss = 0.5 * ori_loss + 0.5 * adv_loss
        else:
            train_loss = adv_loss
        return train_loss, adv_log_vars

    def forward_adv_train_ma2t(
        self,
        ori_loss=None,
        img=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_inds=None,
        l2g_t=None,
        l2g_r_mat=None,
        timestamp=None,
        gt_lane_labels=None,
        gt_lane_bboxes=None,
        gt_lane_masks=None,
        rescale=False,
        gt_fut_traj=None,
        gt_fut_traj_mask=None,
        gt_past_traj=None,
        gt_past_traj_mask=None,
        gt_sdc_bbox=None,
        gt_sdc_label=None,
        gt_sdc_fut_traj=None,
        gt_sdc_fut_traj_mask=None,
        # Occ_gt
        gt_segmentation=None,
        gt_instance=None,
        gt_occ_img_is_valid=None,
        # planning
        sdc_planning=None,
        sdc_planning_mask=None,
        command=None,
        # fut gt for planning
        gt_future_boxes=None,
        **kwargs,  # [1, 9]
    ):
        if self.pgd_cfg.attack_img and self.pgd_cfg.random_start:
            noise_img = torch.empty_like(img).uniform_(
                -self.pgd_cfg.img_eps, self.pgd_cfg.img_eps
            )
            noise_img.requires_grad_(True)
        else:
            noise_img = torch.zeros_like(img)

        if self.pgd_cfg.attack_img:
            noise_img.requires_grad_(True)

        mi_noise_img_grad = 0
        mi_noise_track_grad = 0
        mi_noise_seg_grad = 0
        mi_noise_motion_traj_grad = 0
        mi_noise_motion_sdc_traj_grad = 0

        if (
            self.pgd_cfg.attack_img
            or self.pgd_cfg.attack_track_motion
            or self.pgd_cfg.attack_seg_motion
            or self.pgd_cfg.attack_motion_occ
            or self.pgd_cfg.attack_motion_plan
        ):
            lun = self.pgd_cfg.steps + 1
        else:
            lun = 1

        for pgd_i in range(lun):
            ###### 攻击输入图像 ######
            adv_img = img
            if self.pgd_cfg.attack_img:
                adv_img = img + noise_img

            losses = dict()
            # print(type(img))    # <class 'torch.Tensor'>
            len_queue = adv_img.size(1)  # queue_length
            # print(img.shape)    # torch.Size([1, 3, 6, 3, 928, 1600])
            # print(noise_img.shape)
            # quit()

            ###### 必须深拷贝, forward_train里同理 ######
            track_img_metas = copy.deepcopy(img_metas)  # !!!!!!!!!!!!
            losses_track, outs_track = self.forward_track_train(
                adv_img,
                gt_bboxes_3d,
                gt_labels_3d,
                gt_past_traj,
                gt_past_traj_mask,
                gt_inds,
                gt_sdc_bbox,
                gt_sdc_label,
                l2g_t,
                l2g_r_mat,
                track_img_metas,
                timestamp,
            )
            losses_track = self.loss_weighted_and_prefixed(losses_track, prefix="track")
            track_loss = self.get_task_loss(
                losses_track,
                task_name="track",
                weight=self.ma2t_task_weights[self.task_dict["track"]],
            )
            losses.update(
                {
                    "track.loss": track_loss,
                    "track.weight": self.ma2t_task_weights[self.task_dict["track"]],
                }
            )

            # Upsample bev for tiny version
            outs_track = self.upsample_bev_if_tiny(
                outs_track
            )  # bev_embed 和 bev_pos
            bev_embed = outs_track["bev_embed"]
            bev_pos = outs_track["bev_pos"]

            if 0 == pgd_i:
                assert (
                    outs_track["track_query_embeddings"].shape[-1]
                    == outs_track["sdc_embedding"].shape[-1]
                )
                cat_outs_track = torch.cat(
                    [
                        outs_track["sdc_embedding"][None, :],
                        outs_track["sdc_embedding"][None, :],
                    ],
                    dim=0,
                )
                # print(cat_outs.shape) [2, 256]
                if self.pgd_cfg.attack_track_motion:
                    if self.pgd_cfg.random_start:
                        noise_track = torch.empty_like(cat_outs_track).uniform_(
                            -self.pgd_cfg.track_eps, self.pgd_cfg.track_eps
                        )
                    else:
                        noise_track = torch.zeros_like(cat_outs_track)
                    noise_track.requires_grad_(True)
                else:
                    noise_track = torch.zeros_like(cat_outs_track)
            ############ noise ############
            adv_outs_track = outs_track
            if self.pgd_cfg.attack_track_motion:
                if adv_outs_track["track_query_embeddings"].shape[0] != 0:
                    adv_outs_track["track_query_embeddings"] = adv_outs_track[
                        "track_query_embeddings"
                    ] + torch.cat(
                        [
                            noise_track[0, :][None, :]
                            for _ in range(
                                adv_outs_track["track_query_embeddings"].shape[0]
                            )
                        ],
                        dim=0,
                    )  # noise_track[0:-1, :]
                adv_outs_track["sdc_embedding"] = (
                    adv_outs_track["sdc_embedding"] + noise_track[-1, :]
                )

            seg_img_metas = copy.deepcopy(track_img_metas)  # !!!!!!!!!!!!
            seg_img_metas = [each[len_queue - 1] for each in seg_img_metas]

            outs_seg = dict()
            if self.with_seg_head:
                losses_seg, outs_seg = self.seg_head.forward_train(
                    bev_embed,
                    seg_img_metas,
                    gt_lane_labels,
                    gt_lane_bboxes,
                    gt_lane_masks,
                )
                losses_seg = self.loss_weighted_and_prefixed(losses_seg, prefix="map")
                seg_loss = self.get_task_loss(
                    losses_seg,
                    task_name="map",
                    weight=self.ma2t_task_weights[self.task_dict["map"]],
                )
                losses.update(
                    {
                        "map.loss": seg_loss,
                        "map.weight": self.ma2t_task_weights[self.task_dict["map"]],
                    }
                )
            if 0 == pgd_i:
                assert (
                    outs_seg["args_tuple"][3].shape == outs_seg["args_tuple"][5].shape
                )
                cat_outs_seg = torch.cat(
                    [outs_seg["args_tuple"][3][None], outs_seg["args_tuple"][5][None]],
                    dim=0,
                )
                if self.pgd_cfg.attack_seg_motion:
                    if self.pgd_cfg.random_start:
                        noise_seg = torch.empty_like(cat_outs_seg).uniform_(
                            -self.pgd_cfg.seg_eps, self.pgd_cfg.seg_eps
                        )
                    else:
                        noise_seg = torch.zeros_like(cat_outs_seg)
                    noise_seg.requires_grad_(True)
                else:
                    noise_seg = torch.zeros_like(cat_outs_seg)
            adv_outs_seg = outs_seg
            if self.pgd_cfg.attack_seg_motion:
                adv_outs_seg["args_tuple"][3] = (
                    adv_outs_seg["args_tuple"][3] + noise_seg[0]
                )
                adv_outs_seg["args_tuple"][5] = (
                    adv_outs_seg["args_tuple"][5] + noise_seg[1]
                )

            outs_motion = dict()
            # Forward Motion Head
            if self.with_motion_head:
                ret_dict_motion = self.motion_head.forward_train(
                    bev_embed,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    gt_fut_traj,
                    gt_fut_traj_mask,
                    gt_sdc_fut_traj,
                    gt_sdc_fut_traj_mask,
                    outs_track=adv_outs_track,
                    outs_seg=adv_outs_seg,
                )
                losses_motion = ret_dict_motion["losses"]
                outs_motion = ret_dict_motion["outs_motion"]
                outs_motion["bev_pos"] = bev_pos
                losses_motion = self.loss_weighted_and_prefixed(
                    losses_motion, prefix="motion"
                )
                motion_loss = self.get_task_loss(
                    losses_motion,
                    task_name="motion",
                    weight=self.ma2t_task_weights[self.task_dict["motion"]],
                )
                losses.update(
                    {
                        "motion.loss": motion_loss,
                        "motion.weight": self.ma2t_task_weights[
                            self.task_dict["motion"]
                        ],
                    }
                )

            if self.with_occ_head:
                # assert 0 != outs_motion['track_query'].shape[1]
                if outs_motion["track_query"].shape[1] == 0:  # 是可能出现的
                    # TODO: rm hard code
                    outs_motion["track_query"] = torch.zeros((1, 1, 256)).to(bev_embed)
                    outs_motion["track_query_pos"] = torch.zeros((1, 1, 256)).to(
                        bev_embed
                    )
                    outs_motion["traj_query"] = torch.zeros((3, 1, 1, 6, 256)).to(
                        bev_embed
                    )
                    outs_motion["all_matched_idxes"] = [[-1]]
            if 0 == pgd_i:
                assert "traj_query" in outs_motion.keys()
                assert (
                    outs_motion["track_query"].shape
                    == outs_motion["track_query_pos"].shape
                )
                assert (
                    outs_motion["track_query"].shape[0]
                    == outs_motion["traj_query"].shape[1]
                )
                assert (
                    outs_motion["track_query"].shape[1]
                    == outs_motion["traj_query"].shape[2]
                )
                assert (
                    outs_motion["track_query"].shape[-1]
                    == outs_motion["traj_query"].shape[-1]
                )
                tmp_traj_query = outs_motion["traj_query"][-1]  # 只有 [-1] 被用到了
                tmp_traj_query = tmp_traj_query[:, 0:1, :, :]  # 必须用 0:1 不用能用0
                if self.pgd_cfg.attack_motion_occ:
                    if self.pgd_cfg.random_start:
                        noise_motion_traj = torch.empty_like(tmp_traj_query).uniform_(
                            -self.pgd_cfg.motion_traj_eps, self.pgd_cfg.motion_traj_eps
                        )
                    else:
                        noise_motion_traj = torch.zeros_like(tmp_traj_query)
                    noise_motion_traj.requires_grad_(True)
                else:
                    noise_motion_traj = torch.zeros_like(tmp_traj_query)
            ############ 分别加noise ############
            adv_outs_motion = outs_motion
            if self.pgd_cfg.attack_motion_occ:
                assert "traj_query" in adv_outs_motion.keys()
                # 1, 36, 6, 256
                if outs_motion["track_query"].shape[1] != 0:
                    adv_outs_motion["traj_query"][-1] = adv_outs_motion["traj_query"][
                        -1
                    ] + torch.cat(
                        [
                            noise_motion_traj
                            for _ in range(adv_outs_motion["traj_query"][-1].shape[1])
                        ],
                        dim=1,
                    )

            # Forward Occ Head
            if self.with_occ_head:
                losses_occ = self.occ_head.forward_train(
                    bev_embed,
                    adv_outs_motion,
                    gt_inds_list=gt_inds,
                    gt_segmentation=gt_segmentation,
                    gt_instance=gt_instance,
                    gt_img_is_valid=gt_occ_img_is_valid,
                )
                losses_occ = self.loss_weighted_and_prefixed(losses_occ, prefix="occ")
                occ_loss = self.get_task_loss(
                    losses_occ,
                    task_name="occ",
                    weight=self.ma2t_task_weights[self.task_dict["occ"]],
                )
                losses.update(
                    {
                        "occ.loss": occ_loss,
                        "occ.weight": self.ma2t_task_weights[self.task_dict["occ"]],
                    }
                )

            if 0 == pgd_i:
                # 只用了-1
                tmp_sdc_traj_query = adv_outs_motion["sdc_traj_query"][-1]  # 1,6,256
                if self.pgd_cfg.attack_motion_plan:
                    if self.pgd_cfg.random_start:
                        noise_motion_sdc_traj = torch.empty_like(
                            tmp_sdc_traj_query
                        ).uniform_(
                            -self.pgd_cfg.motion_traj_eps, self.pgd_cfg.motion_traj_eps
                        )
                    else:
                        noise_motion_sdc_traj = torch.zeros_like(tmp_sdc_traj_query)
                    noise_motion_sdc_traj.requires_grad_(True)
                else:
                    noise_motion_sdc_traj = torch.zeros_like(tmp_sdc_traj_query)
            if self.pgd_cfg.attack_motion_plan:
                tmp = [
                    torch.zeros_like(adv_outs_motion["sdc_traj_query"][-1])
                    for _ in range(adv_outs_motion["sdc_traj_query"].shape[0] - 1)
                ]
                adv_outs_motion["sdc_traj_query"] = adv_outs_motion[
                    "sdc_traj_query"
                ] + torch.stack(tmp + [noise_motion_sdc_traj], dim=0)

            # Forward Plan Head
            if self.with_planning_head:
                outs_planning = self.planning_head.forward_train(
                    bev_embed,
                    adv_outs_motion,
                    sdc_planning,
                    sdc_planning_mask,
                    command,
                    gt_future_boxes,
                )
                losses_planning = outs_planning["losses"]
                losses_planning = self.loss_weighted_and_prefixed(
                    losses_planning, prefix="planning"
                )
                plan_loss = self.get_task_loss(
                    losses_planning,
                    task_name="planning",
                    weight=self.ma2t_task_weights[self.task_dict["plan"]],
                )
                losses.update(
                    {
                        "planning.loss": plan_loss,
                        "planning.weight": self.ma2t_task_weights[
                            self.task_dict["plan"]
                        ],
                    }
                )

            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    losses[k] = torch.nan_to_num(v)

            adv_loss, adv_log_vars = self._parse_losses(losses)
            #  mmcv.runner.OptimizerHook
            if pgd_i < self.pgd_cfg.steps:
                if (
                    self.pgd_cfg.attack_img
                    or self.pgd_cfg.attack_track_motion
                    or self.pgd_cfg.attack_seg_motion
                    or self.pgd_cfg.attack_motion_occ
                    or self.pgd_cfg.attack_motion_plan
                ):
                    noises = [
                        noise_img,
                        noise_track,
                        noise_seg,
                        noise_motion_traj,
                        noise_motion_sdc_traj,
                    ]
                    noises_require_grad = list(
                        filter(lambda n: n.requires_grad, noises)
                    )
                    # print(len(noises_require_grad))
                    grads = torch.autograd.grad(
                        outputs=adv_loss,
                        inputs=noises_require_grad,
                        allow_unused=True, 
                    )  
                    # print(noise_img_grad == None)
                index = 0
                if self.pgd_cfg.attack_img:
                    noise_img_grad = grads[index]
                    index = index + 1
                    assert noise_img_grad is not None
                    assert (
                        not noise_img_grad.detach()
                        .cpu()
                        .equal(torch.zeros(noise_img_grad.shape))
                    )
                    mi_noise_img_grad = (
                        self.pgd_cfg.miu * mi_noise_img_grad + noise_img_grad
                    )  # / torch.norm(noise_img_grad, p=2, dim=None, keepdim=False, out=None, dtype=None)
                    noise_img = (
                        noise_img + self.pgd_cfg.img_alpha * mi_noise_img_grad.sign()
                    )
                    # noise_img = torch.clamp(noise_img, min=-self.pgd_cfg.img_eps, max=self.pgd_cfg.img_eps)
                    noise_img = noise_img.detach()
                    noise_img.requires_grad_(True)
                if self.pgd_cfg.attack_track_motion:
                    noise_track_grad = grads[index]
                    index = index + 1
                    assert noise_track_grad is not None
                    assert (
                        not noise_track_grad.detach()
                        .cpu()
                        .equal(torch.zeros(noise_track_grad.shape))
                    )
                    mi_noise_track_grad = (
                        self.pgd_cfg.miu * mi_noise_track_grad + noise_track_grad
                    )  
                    noise_track = (
                        noise_track
                        + self.pgd_cfg.track_alpha * mi_noise_track_grad.sign()
                    )
                    noise_track = noise_track.detach()
                    noise_track.requires_grad_(True)
                if self.pgd_cfg.attack_seg_motion:
                    noise_seg_grad = grads[index]
                    index = index + 1
                    assert noise_seg_grad is not None
                    assert (
                        not noise_seg_grad.detach()
                        .cpu()
                        .equal(torch.zeros(noise_seg_grad.shape))
                    )
                    mi_noise_seg_grad = (
                        self.pgd_cfg.miu * mi_noise_seg_grad + noise_seg_grad
                    )  
                    noise_seg = (
                        noise_seg + self.pgd_cfg.seg_alpha * mi_noise_seg_grad.sign()
                    )
                    noise_seg = noise_seg.detach()
                    noise_seg.requires_grad_(True)
                if self.pgd_cfg.attack_motion_occ:
                    noise_motion_traj_grad = grads[index]
                    index = index + 1
                    assert noise_motion_traj_grad is not None
                    mi_noise_motion_traj_grad = (
                        self.pgd_cfg.miu * mi_noise_motion_traj_grad
                        + noise_motion_traj_grad
                    )  
                    noise_motion_traj = (
                        noise_motion_traj
                        + self.pgd_cfg.motion_traj_alpha
                        * mi_noise_motion_traj_grad.sign()
                    )
                    noise_motion_traj = noise_motion_traj.detach()
                    noise_motion_traj.requires_grad_(True)
                if self.pgd_cfg.attack_motion_plan:
                    noise_motion_sdc_traj_grad = grads[index]
                    index = index + 1
                    assert noise_motion_sdc_traj_grad is not None
                    mi_noise_motion_sdc_traj_grad = (
                        self.pgd_cfg.miu * mi_noise_motion_sdc_traj_grad
                        + noise_motion_sdc_traj_grad
                    )  
                    noise_motion_sdc_traj = (
                        noise_motion_sdc_traj
                        + self.pgd_cfg.motion_traj_alpha
                        * mi_noise_motion_sdc_traj_grad.sign()
                    )
                    noise_motion_sdc_traj = noise_motion_sdc_traj.detach()
                    noise_motion_sdc_traj.requires_grad_(True)
            if pgd_i == self.pgd_cfg.steps - 1:
                noise_motion_sdc_traj = noise_motion_sdc_traj.clone().detach()
                noise_motion_sdc_traj.requires_grad_(False)
            # quit()
            # optimizer zero_grad 
        del noise_motion_sdc_traj
        # print(noise_img.requires_grad)
        # print(noise_motion_sdc_traj.requires_grad)
        # noise_motion_sdc_traj.requires_grad_(False)

        if ori_loss is not None:
            train_loss = 0.5 * ori_loss + 0.5 * adv_loss
        else:
            train_loss = adv_loss
        return train_loss, adv_log_vars

    # track的attack改用了test的模板
    def forward_attack_v1_task_loss_res(
        self,
        img=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_inds=None,
        l2g_t=None,
        l2g_r_mat=None,
        timestamp=None,
        gt_lane_labels=None,
        gt_lane_bboxes=None,
        gt_lane_masks=None,
        rescale=False,
        gt_fut_traj=None,
        gt_fut_traj_mask=None,
        gt_past_traj=None,
        gt_past_traj_mask=None,
        gt_sdc_bbox=None,
        gt_sdc_label=None,
        gt_sdc_fut_traj=None,
        gt_sdc_fut_traj_mask=None,
        # Occ_gt
        gt_segmentation=None,
        gt_instance=None,
        gt_occ_img_is_valid=None,
        # planning
        sdc_planning=None,
        sdc_planning_mask=None,
        command=None,
        # fut gt for planning
        gt_future_boxes=None,
        **kwargs,  # [1, 9]
    ):
        for param in self.parameters():
            param.requires_grad = False
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            # the first sample of each scene is truncated
            self.prev_frame_info["prev_bev"] = None
        # update idx
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])
        # first frame
        if self.prev_frame_info["scene_token"] is None:
            img_metas[0][0]["can_bus"][:3] = 0
            img_metas[0][0]["can_bus"][-1] = 0
        # following frames
        else:
            img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle

        if isinstance(img, list):
            img = img[0]
        if img.dim() != 5:
            # bs, nq, n, C, H, W -> bs, n, C, H, W
            img = torch.squeeze(img, dim=1)
        if isinstance(img_metas[0], list):  # test
            img_metas = img_metas[0]
        elif isinstance(img_metas[0], dict):  # train
            img_metas[0] = img_metas[0][0]
        timestamp = timestamp[0]
        if isinstance(timestamp, list):
            timestamp = timestamp[0] if timestamp is not None else None
        import mmcv

        result = [dict()]

        adv_img = img
        if self.pgd_cfg.attack_img:
            noise_img = torch.empty_like(img).uniform_(
                -self.pgd_cfg.img_eps, self.pgd_cfg.img_eps
            )
            noise_img.requires_grad_(True)
            # print(img.shape, noise_img.shape)   torch.Size([1, 6, 3, 928, 1600]) torch.Size([1, 1, 6, 3, 928, 1600])
        else:
            noise_img = torch.zeros_like(img)
        if self.pgd_cfg.attack_img:
            lun = self.pgd_cfg.steps + 1
        else:
            lun = 1
        for pgd_i in range(lun):
            if pgd_i == (lun - 1):
                noise_img = noise_img.detach()
                noise_img.requires_grad_(False)
                with torch.no_grad():
                    adv_img = img + noise_img
                    track_img_metas = copy.deepcopy(img_metas)
                    losses_track, outs_track = self.simple_attack_track(
                        adv_img,
                        gt_bboxes_3d,
                        gt_labels_3d,
                        gt_past_traj,
                        gt_past_traj_mask,
                        gt_inds,
                        gt_sdc_bbox,
                        gt_sdc_label,
                        l2g_t[0][0].unsqueeze(0),
                        l2g_r_mat[0][0].unsqueeze(0),
                        track_img_metas,
                        timestamp,
                    )
                break

            adv_img = img + noise_img
            track_img_metas = copy.deepcopy(img_metas)
            losses_track, outs_track = self.simple_attack_track(
                adv_img,
                gt_bboxes_3d,
                gt_labels_3d,
                gt_past_traj,
                gt_past_traj_mask,
                gt_inds,
                gt_sdc_bbox,
                gt_sdc_label,
                l2g_t[0][0].unsqueeze(0),
                l2g_r_mat[0][0].unsqueeze(0),
                track_img_metas,
                timestamp,
            )
            losses_track = self.loss_weighted_and_prefixed(losses_track, prefix="track")
            track_loss = self.get_task_loss(losses_track, "track")
            if self.pgd_cfg.attack_img:
                if pgd_i < (lun - 1):
                    noise_img_grad = torch.autograd.grad(
                        outputs=track_loss, inputs=noise_img
                    )[0]
                    assert noise_img_grad is not None
                    assert (
                        not noise_img_grad.detach()
                        .cpu()
                        .equal(torch.zeros(noise_img_grad.shape))
                    )
                    noise_img = noise_img + self.pgd_cfg.img_alpha * noise_img_grad.sign()
                    noise_img = noise_img.detach()
                if pgd_i < (lun - 2):
                    noise_img.requires_grad_(True)
                else:
                    noise_img.requires_grad_(False)

        # print("\nMemory Allocated:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")
        # Upsample bev for tiny version
        outs_track = self.upsample_bev_if_tiny(
            outs_track
        ) 
        result_track = [{} for _ in range(img.shape[0])]
        result_track[0] = outs_track
        bev_embed = outs_track["bev_embed"]
        bev_pos = outs_track["bev_pos"]

        seg_img_metas = copy.deepcopy(track_img_metas)  # !!!!!!!!!!!!
        # seg_img_metas = [each[len_queue-1] for each in seg_img_metas]
        outs_seg = dict()
        if self.with_seg_head:
            losses_seg, outs_seg, result_seg = self.seg_head.forward_attack(
                bev_embed, seg_img_metas, gt_lane_labels, gt_lane_bboxes, gt_lane_masks
            )
            losses_seg = self.loss_weighted_and_prefixed(losses_seg, prefix="map")
        
        if self.pgd_cfg.attack_track_motion or self.pgd_cfg.attack_seg_motion:
            lun = self.pgd_cfg.steps + 1
        else:
            lun = 1
        for pgd_i in range(lun):
            # outs_track.keys()
            # dict_keys(['bev_embed', 'bev_pos', 'track_query_embeddings', 'track_query_matched_idxes', 'track_bbox_results', 'sdc_boxes_3d', 'sdc_scores_3d', 'sdc_track_scores', 'sdc_track_bbox_results', 'sdc_embedding'])
            # print(outs_track['track_query_embeddings'].shape)    # [37, 256]  [39, 256]  大小不固定
            # print(outs_track['sdc_embedding'].shape)
            
            if 0 == pgd_i:
                assert (
                    outs_track["track_query_embeddings"].shape[-1]
                    == outs_track["sdc_embedding"].shape[-1]
                )
                cat_outs_track = torch.cat(
                    [
                        outs_track["sdc_embedding"][None, :],
                        outs_track["sdc_embedding"][None, :],
                    ],
                    dim=0,
                )
                # print(cat_outs.shape) [2, 256]
                if self.pgd_cfg.attack_track_motion:
                    if self.pgd_cfg.random_start:
                        noise_track = torch.empty_like(cat_outs_track).uniform_(
                            -self.pgd_cfg.track_eps, self.pgd_cfg.track_eps
                        )
                    else:
                        noise_track = torch.zeros_like(cat_outs_track)
                    noise_track.requires_grad_(True)
                else:
                    noise_track = torch.zeros_like(cat_outs_track)

            for k, v in outs_track.items():
                if isinstance(v, torch.Tensor):
                    outs_track[k] = v.detach()
                elif isinstance(v, list):
                    for lv in v:
                        if isinstance(lv, torch.Tensor):
                            lv = lv.detach()
            adv_outs_track = copy.deepcopy(outs_track)
            
            # else:
            #     adv_outs_track = copy.deepcopy(outs_track)
            if self.pgd_cfg.attack_track_motion:
                # print(noise_track[0:-1, :].shape)
                # print(noise_track[-1, :].shape)
                # print(adv_outs_track['track_query_embeddings'].shape)
                if adv_outs_track["track_query_embeddings"].shape[0] != 0:
                    # Qa
                    adv_outs_track["track_query_embeddings"] = adv_outs_track[
                        "track_query_embeddings"
                    ] + torch.cat(
                        [
                            noise_track[0, :][None, :]
                            for _ in range(
                                adv_outs_track["track_query_embeddings"].shape[0]
                            )
                        ],
                        dim=0,
                    )  # noise_track[0:-1, :]
                adv_outs_track["sdc_embedding"] = (
                    adv_outs_track["sdc_embedding"] + noise_track[-1, :]
                )
                
            if 0 == pgd_i:
                assert (
                    outs_seg["args_tuple"][3].shape == outs_seg["args_tuple"][5].shape
                )
                cat_outs_seg = torch.cat(
                    [outs_seg["args_tuple"][3][None], outs_seg["args_tuple"][5][None]],
                    dim=0,
                )
                if self.pgd_cfg.attack_seg_motion:
                    if self.pgd_cfg.random_start:
                        noise_seg = torch.empty_like(cat_outs_seg).uniform_(
                            -self.pgd_cfg.seg_eps, self.pgd_cfg.seg_eps
                        )
                    else:
                        noise_seg = torch.zeros_like(cat_outs_seg)
                    noise_seg.requires_grad_(True)
                else:
                    noise_seg = torch.zeros_like(cat_outs_seg)

            # if lun == 1:
            for k, v in outs_seg.items():
                if isinstance(v, torch.Tensor):
                    outs_seg[k] = v.detach()
            adv_outs_seg = copy.deepcopy(outs_seg)
            # else:
            #     adv_outs_seg = copy.deepcopy(outs_seg)
            if self.pgd_cfg.attack_seg_motion:
                # Qm (300, 256)
                adv_outs_seg["args_tuple"][3] = (
                    adv_outs_seg["args_tuple"][3] + noise_seg[0]
                )
                adv_outs_seg["args_tuple"][5] = (
                    adv_outs_seg["args_tuple"][5] + noise_seg[1]
                )

            outs_motion = dict()
            if self.with_motion_head:
                ret_dict_motion, result_motion = self.motion_head.forward_attack(
                    bev_embed,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    gt_fut_traj,
                    gt_fut_traj_mask,
                    gt_sdc_fut_traj,
                    gt_sdc_fut_traj_mask,
                    outs_track=adv_outs_track,
                    outs_seg=adv_outs_seg,
                )
                losses_motion = ret_dict_motion["losses"]
                outs_motion = ret_dict_motion["outs_motion"]
                outs_motion["bev_pos"] = bev_pos
                if self.pgd_cfg.attack_track_motion or self.pgd_cfg.attack_seg_motion:
                    losses_motion = self.loss_weighted_and_prefixed(
                        losses_motion, prefix="motion"
                    )
                    motion_loss = self.get_task_loss(losses_motion, "motion")
                    noises = [noise_track, noise_seg]
                    noises_require_grad = list(
                        filter(lambda n: n.requires_grad, noises)
                    )
                    grads = torch.autograd.grad(
                        outputs=motion_loss,
                        inputs=noises_require_grad,
                        allow_unused=True,  
                    ) 
                    index = 0
                    if self.pgd_cfg.attack_track_motion:
                        noise_track_grad = grads[index]
                        index = index + 1
                        assert noise_track_grad is not None
                        assert (
                            not noise_track_grad.detach()
                            .cpu()
                            .equal(torch.zeros(noise_track_grad.shape))
                        )
                        # print(noise_track_grad)
                        noise_track = (
                            noise_track
                            + self.pgd_cfg.track_alpha * noise_track_grad.sign()
                        )
                        del noise_track_grad
                        noise_track = noise_track.detach()
                        if pgd_i < (lun - 1):
                            noise_track.requires_grad_(True)

                    if self.pgd_cfg.attack_seg_motion:
                        noise_seg_grad = grads[index]
                        index = index + 1
                        assert noise_seg_grad is not None
                        assert (
                            not noise_seg_grad.detach()
                            .cpu()
                            .equal(torch.zeros(noise_seg_grad.shape))
                        )
                        noise_seg = (
                            noise_seg + self.pgd_cfg.seg_alpha * noise_seg_grad.sign()
                        )
                        del noise_seg_grad
                        noise_seg = noise_seg.detach()
                        if pgd_i < (lun - 1):
                            noise_seg.requires_grad_(True)
        occ_no_query = outs_motion["track_query"].shape[1] == 0
        if outs_motion["track_query"].shape[1] == 0:  # 可能出现的
            # TODO: rm hard code
            # 这部分必须移到循环外面
            outs_motion["track_query"] = torch.zeros((1, 1, 256)).to(bev_embed)
            outs_motion["track_query_pos"] = torch.zeros((1, 1, 256)).to(bev_embed)
            outs_motion["traj_query"] = torch.zeros((3, 1, 1, 6, 256)).to(bev_embed)
            outs_motion["all_matched_idxes"] = [[-1]]

        if self.pgd_cfg.attack_motion_occ:
            lun = self.pgd_cfg.steps + 1
        else:
            lun = 1
        for pgd_i in range(lun):
            if 0 == pgd_i:
                assert "traj_query" in outs_motion.keys()
                assert (
                    outs_motion["track_query"].shape
                    == outs_motion["track_query_pos"].shape
                )
                assert (
                    outs_motion["track_query"].shape[0]
                    == outs_motion["traj_query"].shape[1]
                )
                assert (
                    outs_motion["track_query"].shape[1]
                    == outs_motion["traj_query"].shape[2]
                )
                assert (
                    outs_motion["track_query"].shape[-1]
                    == outs_motion["traj_query"].shape[-1]
                )
                tmp_traj_query = outs_motion["traj_query"][-1]  # 只有 [-1] 被occ用到了
                tmp_traj_query = tmp_traj_query[:, 0:1, :, :]  # 必须用 0:1 不用能用0
                tmp_track_query = outs_motion["track_query"][None, :, 0:1, :]
                # track_query_pos 后面detach了  不要加noise!!!!!!
                # tmp_track_query_pos = outs_motion['track_query_pos'][None, :, 0:1, :]
                # cat_outs_motion_track = torch.cat([tmp_track_query, tmp_track_query_pos], dim=0)
                if self.pgd_cfg.attack_motion_occ:
                    if self.pgd_cfg.random_start:
                        noise_motion_traj = torch.empty_like(tmp_traj_query).uniform_(
                            -self.pgd_cfg.motion_traj_eps, self.pgd_cfg.motion_traj_eps
                        )
                        # plan_new_change noise_motion_track = torch.empty_like(tmp_track_query).uniform_(-self.pgd_cfg.motion_track_eps, self.pgd_cfg.motion_track_eps)
                    else:
                        noise_motion_traj = torch.zeros_like(tmp_traj_query)
                        # plan_new_change noise_motion_track = torch.zeros_like(tmp_track_query)
                    noise_motion_traj.requires_grad_(True)
                    # plan_new_change noise_motion_track.requires_grad_(True)
                else:
                    noise_motion_traj = torch.zeros_like(tmp_traj_query)
                    # plan_new_change noise_motion_track = torch.zeros_like(tmp_track_query)

            ############ 分别加noise ############
            if lun == 1:
                adv_outs_motion = outs_motion
            else:
                adv_outs_motion = outs_motion  # copy.deepcopy(outs_motion)
            if self.pgd_cfg.attack_motion_occ:
                assert "traj_query" in adv_outs_motion.keys()
                # 1, 36, 6, 256
                if outs_motion["track_query"].shape[1] != 0:
                    adv_outs_motion["traj_query"][-1] = adv_outs_motion["traj_query"][
                        -1
                    ] + torch.cat(
                        [
                            noise_motion_traj
                            for _ in range(adv_outs_motion["traj_query"][-1].shape[1])
                        ],
                        dim=1,
                    )  # noise_motion[0:-1, :]
                    # plan_new_change adv_outs_motion['track_query'] = adv_outs_motion['track_query'] + torch.cat([noise_motion_track[0] for _ in range(outs_motion['track_query'].shape[1])], dim=1)
                else:
                    print("motion occ   00000")

            # Forward Occ Head
            if self.with_occ_head:
                losses_occ, outs_occ = self.occ_head.forward_attack(
                    bev_feat=bev_embed,
                    outs_dict=adv_outs_motion,
                    no_query=occ_no_query,
                    gt_inds_list=copy.deepcopy(gt_inds),
                    gt_segmentation=copy.deepcopy(gt_segmentation),
                    gt_instance=copy.deepcopy(gt_instance),
                    gt_img_is_valid=copy.deepcopy(gt_occ_img_is_valid),
                )
                losses_occ = self.loss_weighted_and_prefixed(losses_occ, prefix="occ")
                result[0]["occ"] = outs_occ
                if self.pgd_cfg.attack_motion_occ:
                    occ_loss = self.get_task_loss(losses_occ, "occ")
                    noise_motion_traj_grad = torch.autograd.grad(
                        outputs=occ_loss,
                        # plan_new_change inputs=[noise_motion_traj, noise_motion_track],
                        inputs=noise_motion_traj,
                    )[0]
                    assert noise_motion_traj_grad is not None
                    noise_motion_traj = (
                        noise_motion_traj
                        + self.pgd_cfg.motion_traj_alpha * noise_motion_traj_grad.sign()
                    )
                    del noise_motion_traj_grad
                    noise_motion_traj = noise_motion_traj.detach()
                    if pgd_i < (lun - 1):
                        noise_motion_traj.requires_grad_(True)
                    # plan_new_change del noise_motion_track_grad

        if self.pgd_cfg.attack_motion_plan:
            lun = self.pgd_cfg.steps + 1
        else:
            lun = 1
        for pgd_i in range(lun):
            if lun == 1:
                adv_outs_motion = outs_motion
            else:
                adv_outs_motion = outs_motion 
            if 0 == pgd_i:
                # 只用了-1
                tmp_sdc_traj_query = adv_outs_motion["sdc_traj_query"][-1]  # 1,6,256
                if self.pgd_cfg.attack_motion_plan:
                    if self.pgd_cfg.random_start:
                        noise_motion_sdc_traj = torch.empty_like(
                            tmp_sdc_traj_query
                        ).uniform_(
                            -self.pgd_cfg.motion_traj_eps, self.pgd_cfg.motion_traj_eps
                        )
                    else:
                        noise_motion_sdc_traj = torch.zeros_like(tmp_sdc_traj_query)
                    noise_motion_sdc_traj.requires_grad_(True)
                else:
                    noise_motion_sdc_traj = torch.zeros_like(tmp_sdc_traj_query)
            if self.pgd_cfg.attack_motion_plan:
                # 1, 6, 256
                adv_outs_motion["sdc_traj_query"][-1] = (
                    adv_outs_motion["sdc_traj_query"][-1] + noise_motion_sdc_traj
                )

            # Forward Plan Head
            if self.with_planning_head:
                outs_planning, result_planning = self.planning_head.forward_attack(
                    bev_embed=bev_embed,
                    outs_motion=adv_outs_motion,
                    outs_occflow=outs_occ,
                    command=command,
                    sdc_planning=copy.deepcopy(sdc_planning),
                    sdc_planning_mask=copy.deepcopy(sdc_planning_mask),
                    gt_future_boxes=copy.deepcopy(gt_future_boxes),
                )
                losses_planning = outs_planning["losses"]
                losses_planning = self.loss_weighted_and_prefixed(
                    losses_planning, prefix="planning"
                )
                planning_gt = dict(
                    segmentation=gt_segmentation,
                    sdc_planning=sdc_planning,
                    sdc_planning_mask=sdc_planning_mask,
                    command=command,
                )
                result[0]["planning"] = dict(
                    planning_gt=planning_gt,
                    result_planning=result_planning,
                )
                if self.pgd_cfg.attack_motion_plan:
                    plan_loss = self.get_task_loss(losses_planning, "planning")
                    noise_motion_sdc_traj_grad = torch.autograd.grad(
                        outputs=plan_loss,
                        inputs=noise_motion_sdc_traj,
                    )[
                        0
                    ] 
                    assert noise_motion_sdc_traj_grad is not None
                    noise_motion_sdc_traj = (
                        noise_motion_sdc_traj
                        + self.pgd_cfg.motion_traj_alpha
                        * noise_motion_sdc_traj_grad.sign()
                    )
                    noise_motion_sdc_traj = noise_motion_sdc_traj.detach()
                    if pgd_i < (lun - 1):
                        noise_motion_sdc_traj.requires_grad_(True)
                    
        del tmp_sdc_traj_query, tmp_track_query, tmp_traj_query
        torch.cuda.empty_cache()
        result_track = [outs_track]
        pop_track_list = [
            "prev_bev",
            "bev_pos",
            "bev_embed",
            "track_query_embeddings",
            "sdc_embedding",
        ]
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)

        if self.with_seg_head:
            result_seg[0] = pop_elem_in_result(
                result_seg[0], pop_list=["args_tuple", "pts_bbox"]
            )  # 'pts_bbox',用于vis地图
        if self.with_motion_head:
            result_motion[0] = pop_elem_in_result(result_motion[0])
        if self.with_occ_head:
            result[0]["occ"] = pop_elem_in_result(
                result[0]["occ"],
                pop_list=[
                    "seg_out_mask",
                    "flow_out",
                    "future_states_occ",
                    "pred_ins_masks",
                    "pred_raw_occ",
                    "pred_ins_logits",
                    "pred_ins_sigmoid",
                ],
            )

        # for param in self.parameters():
        #     param = param.detach()
        for i, res in enumerate(result):
            # res['token'] = img_metas[i][len_queue-1]['sample_idx']
            res["token"] = img_metas[i]["sample_idx"]
            res.update(result_track[i])
            if self.with_motion_head:
                res.update(result_motion[i])
            if self.with_seg_head:
                res.update(result_seg[i])
        for i, res in enumerate(result):
            for k, v in res.items():
                if isinstance(v, torch.Tensor):
                    res[k] = v.detach()  # .cpu()
                elif isinstance(v, dict):
                    for dk, dv in v.items():
                        if isinstance(dv, torch.Tensor):
                            res[k][dk] = dv.detach()  # .cpu()
                        elif isinstance(dv, dict):
                            for ddk, ddv in dv.items():
                                if isinstance(ddv, torch.Tensor):
                                    res[k][dk][ddk] = ddv.detach()  # .cpu()
                elif isinstance(v, list):
                    for li, lv in enumerate(v):
                        if isinstance(lv, list):
                            for lli, llv in enumerate(lv):
                                if isinstance(llv, torch.Tensor):
                                    res[k][li][lli] = llv.detach()
                                elif isinstance(llv, LiDARInstance3DBoxes):
                                    llv.tensor = llv.tensor.detach()
                        # else:
                        #     print('list for3', type(dv))
                elif isinstance(v, LiDARInstance3DBoxes):
                    v.tensor = v.tensor.detach()
        return result

    def forward_attack_plan_loss_res(    
        self,
        img=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_inds=None,
        l2g_t=None,
        l2g_r_mat=None,
        timestamp=None,
        gt_lane_labels=None,
        gt_lane_bboxes=None,
        gt_lane_masks=None,
        rescale=False,
        gt_fut_traj=None,
        gt_fut_traj_mask=None,
        gt_past_traj=None,
        gt_past_traj_mask=None,
        gt_sdc_bbox=None,
        gt_sdc_label=None,
        gt_sdc_fut_traj=None,
        gt_sdc_fut_traj_mask=None,
        # Occ_gt
        gt_segmentation=None,
        gt_instance=None,
        gt_occ_img_is_valid=None,
        # planning
        sdc_planning=None,
        sdc_planning_mask=None,
        command=None,
        # fut gt for planning
        gt_future_boxes=None,
        **kwargs,  # [1, 9]
    ):
        meta_data = img_metas[0][0]

        # 提取 filename
        file_names = meta_data["filename"]

        # 打印 filename
        for param in self.parameters():
            param.requires_grad = False
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            # the first sample of each scene is truncated
            self.prev_frame_info["prev_bev"] = None
        # update idx
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])
        # first frame
        if self.prev_frame_info["scene_token"] is None:
            img_metas[0][0]["can_bus"][:3] = 0
            img_metas[0][0]["can_bus"][-1] = 0
        # following frames
        else:
            img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle

        if isinstance(img, list):
            img = img[0]
        if img.dim() != 5:
            # bs, nq, n, C, H, W -> bs, n, C, H, W
            img = torch.squeeze(img, dim=1)
        if isinstance(img_metas[0], list):  # test
            img_metas = img_metas[0]
        elif isinstance(img_metas[0], dict):  # train
            img_metas[0] = img_metas[0][0]
        timestamp = timestamp[0]
        if isinstance(timestamp, list):
            timestamp = timestamp[0] if timestamp is not None else None
        result = [dict()]

        if (
            self.pgd_cfg.attack_img
            or self.pgd_cfg.attack_track_motion
            or self.pgd_cfg.attack_seg_motion
            or self.pgd_cfg.attack_motion_occ
            or self.pgd_cfg.attack_motion_plan
        ):
            lun = self.pgd_cfg.steps + 1
        else:
            lun = 1

        for pgd_i in range(lun):
            losses = dict()

            if pgd_i == 0:
                if self.pgd_cfg.attack_img and self.pgd_cfg.random_start:
                    noise_img = torch.empty_like(img).uniform_(
                        -self.pgd_cfg.img_eps, self.pgd_cfg.img_eps
                    )
                else:
                    noise_img = torch.zeros_like(img)
                if self.pgd_cfg.attack_img:
                    noise_img.requires_grad_(True)
            adv_img = img
            if self.pgd_cfg.attack_img:
                adv_img = img + noise_img

            track_img_metas = copy.deepcopy(img_metas)  # !!!!!!!!!!!!
            losses_track, outs_track = self.simple_attack_track(
                adv_img,
                gt_bboxes_3d,
                gt_labels_3d,
                gt_past_traj,
                gt_past_traj_mask,
                gt_inds,
                gt_sdc_bbox,
                gt_sdc_label,
                l2g_t[0][0].unsqueeze(0),
                l2g_r_mat[0][0].unsqueeze(0),
                track_img_metas,
                timestamp,
            )
            losses_track = self.loss_weighted_and_prefixed(losses_track, prefix="track")
            losses.update(losses_track)
            # print(losses_track[list(losses_track.keys())[0]])

            # Upsample bev for tiny version
            outs_track = self.upsample_bev_if_tiny(
                outs_track
            ) 
            result_track = [{} for _ in range(img.shape[0])]
            result_track[0] = outs_track
            bev_embed = outs_track["bev_embed"]
            bev_pos = outs_track["bev_pos"]


            seg_img_metas = copy.deepcopy(track_img_metas)  # !!!!!!!!!!!!
            outs_seg = dict()
            if self.with_seg_head:
                losses_seg, outs_seg, result_seg = self.seg_head.forward_attack(
                    bev_embed,
                    seg_img_metas,
                    gt_lane_labels,
                    gt_lane_bboxes,
                    gt_lane_masks,
                )
                losses_seg = self.loss_weighted_and_prefixed(losses_seg, prefix="map")
                losses.update(losses_seg)
            if 0 == pgd_i:
                assert (
                    outs_track["track_query_embeddings"].shape[-1]
                    == outs_track["sdc_embedding"].shape[-1]
                )
                cat_outs_track = torch.cat(
                    [
                        outs_track["sdc_embedding"][None, :],
                        outs_track["sdc_embedding"][None, :],
                    ],
                    dim=0,
                )
                # print(cat_outs.shape) [2, 256]
                if self.pgd_cfg.attack_track_motion:
                    if self.pgd_cfg.random_start:
                        noise_track = torch.empty_like(cat_outs_track).uniform_(
                            -self.pgd_cfg.track_eps, self.pgd_cfg.track_eps
                        )
                    else:
                        noise_track = torch.zeros_like(cat_outs_track)
                    noise_track.requires_grad_(True)
                else:
                    noise_track = torch.zeros_like(cat_outs_track)

            adv_outs_track = outs_track
            if self.pgd_cfg.attack_track_motion:
                if adv_outs_track["track_query_embeddings"].shape[0] != 0:
                    adv_outs_track["track_query_embeddings"] = adv_outs_track[
                        "track_query_embeddings"
                    ] + torch.cat(
                        [
                            noise_track[0, :][None, :]
                            for _ in range(
                                adv_outs_track["track_query_embeddings"].shape[0]
                            )
                        ],
                        dim=0,
                    )  # noise_track[0:-1, :]
                adv_outs_track["sdc_embedding"] = (
                    adv_outs_track["sdc_embedding"] + noise_track[-1, :]
                )

            if 0 == pgd_i:
                assert (
                    outs_seg["args_tuple"][3].shape == outs_seg["args_tuple"][5].shape
                )
                cat_outs_seg = torch.cat(
                    [outs_seg["args_tuple"][3][None], outs_seg["args_tuple"][5][None]],
                    dim=0,
                )
                if self.pgd_cfg.attack_seg_motion:
                    if self.pgd_cfg.random_start:
                        noise_seg = torch.empty_like(cat_outs_seg).uniform_(
                            -self.pgd_cfg.seg_eps, self.pgd_cfg.seg_eps
                        )
                    else:
                        noise_seg = torch.zeros_like(cat_outs_seg)
                    noise_seg.requires_grad_(True)
                else:
                    noise_seg = torch.zeros_like(cat_outs_seg)

            adv_outs_seg = outs_seg
            if self.pgd_cfg.attack_seg_motion:
                adv_outs_seg["args_tuple"][3] = (
                    adv_outs_seg["args_tuple"][3] + noise_seg[0]
                )
                adv_outs_seg["args_tuple"][5] = (
                    adv_outs_seg["args_tuple"][5] + noise_seg[1]
                )

            outs_motion = dict()
            # Forward Motion Head
            if self.with_motion_head:
                ret_dict_motion, result_motion = self.motion_head.forward_attack(
                    bev_embed,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    gt_fut_traj,
                    gt_fut_traj_mask,
                    gt_sdc_fut_traj,
                    gt_sdc_fut_traj_mask,
                    outs_track=adv_outs_track,
                    outs_seg=adv_outs_seg,
                )
                losses_motion = ret_dict_motion["losses"]
                outs_motion = ret_dict_motion["outs_motion"]
                outs_motion["bev_pos"] = bev_pos
                losses_motion = self.loss_weighted_and_prefixed(
                    losses_motion, prefix="motion"
                )
                losses.update(losses_motion)

            # fix 0203
            if self.with_occ_head:
                occ_no_query = outs_motion["track_query"].shape[1] == 0
                if outs_motion["track_query"].shape[1] == 0: 
                    # TODO: rm hard code
                    outs_motion["track_query"] = torch.zeros((1, 1, 256)).to(bev_embed)
                    outs_motion["track_query_pos"] = torch.zeros((1, 1, 256)).to(
                        bev_embed
                    )
                    outs_motion["traj_query"] = torch.zeros((3, 1, 1, 6, 256)).to(
                        bev_embed
                    )
                    outs_motion["all_matched_idxes"] = [[-1]]
            if 0 == pgd_i:
                assert "traj_query" in outs_motion.keys()
                assert (
                    outs_motion["track_query"].shape
                    == outs_motion["track_query_pos"].shape
                )
                assert (
                    outs_motion["track_query"].shape[0]
                    == outs_motion["traj_query"].shape[1]
                )
                assert (
                    outs_motion["track_query"].shape[1]
                    == outs_motion["traj_query"].shape[2]
                )
                assert (
                    outs_motion["track_query"].shape[-1]
                    == outs_motion["traj_query"].shape[-1]
                )
                tmp_traj_query = outs_motion["traj_query"][-1]  # 只有 [-1] 被occ用到了
                tmp_traj_query = tmp_traj_query[:, 0:1, :, :]  # 必须用 0:1 不用能用0
                tmp_track_query = outs_motion["track_query"][None, :, 0:1, :]
                if self.pgd_cfg.attack_motion_occ:
                    if self.pgd_cfg.random_start:
                        noise_motion_traj = torch.empty_like(tmp_traj_query).uniform_(
                            -self.pgd_cfg.motion_traj_eps, self.pgd_cfg.motion_traj_eps
                        )
                        
                    else:
                        noise_motion_traj = torch.zeros_like(tmp_traj_query)
                    noise_motion_traj.requires_grad_(True)
                else:
                    noise_motion_traj = torch.zeros_like(tmp_traj_query)

            adv_outs_motion = outs_motion
            if self.pgd_cfg.attack_motion_occ:
                assert "traj_query" in adv_outs_motion.keys()
                # 1, 36, 6, 256
                if outs_motion["track_query"].shape[1] != 0:
                    adv_outs_motion["traj_query"][-1] = adv_outs_motion["traj_query"][
                        -1
                    ] + torch.cat(
                        [
                            noise_motion_traj
                            for _ in range(adv_outs_motion["traj_query"][-1].shape[1])
                        ],
                        dim=1,
                    )  
                else:
                    print("motion occ   00000")

            # Forward Occ Head
            if self.with_occ_head:
                losses_occ, outs_occ = self.occ_head.forward_attack(
                    bev_feat=bev_embed,
                    outs_dict=adv_outs_motion,
                    no_query=occ_no_query,
                    gt_inds_list=copy.deepcopy(gt_inds),
                    gt_segmentation=copy.deepcopy(gt_segmentation),
                    gt_instance=copy.deepcopy(gt_instance),
                    gt_img_is_valid=copy.deepcopy(gt_occ_img_is_valid),
                )
                losses_occ = self.loss_weighted_and_prefixed(losses_occ, prefix="occ")
                losses.update(losses_occ)
                result[0]["occ"] = outs_occ

            adv_outs_motion = outs_motion
            if 0 == pgd_i:
                # 只用了-1
                tmp_sdc_traj_query = adv_outs_motion["sdc_traj_query"][-1]  # 1,6,256
                if self.pgd_cfg.attack_motion_plan:
                    if self.pgd_cfg.random_start:
                        noise_motion_sdc_traj = torch.empty_like(
                            tmp_sdc_traj_query
                        ).uniform_(
                            -self.pgd_cfg.motion_traj_eps, self.pgd_cfg.motion_traj_eps
                        )
                    else:
                        noise_motion_sdc_traj = torch.zeros_like(tmp_sdc_traj_query)
                    noise_motion_sdc_traj.requires_grad_(True)
                else:
                    noise_motion_sdc_traj = torch.zeros_like(tmp_sdc_traj_query)
            if self.pgd_cfg.attack_motion_plan:
                # 1, 6, 256
                adv_outs_motion["sdc_traj_query"][-1] = (
                    adv_outs_motion["sdc_traj_query"][-1] + noise_motion_sdc_traj
                )

            # Forward Plan Head
            if self.with_planning_head:
                outs_planning, result_planning = self.planning_head.forward_attack(
                    bev_embed=bev_embed,
                    outs_motion=adv_outs_motion,
                    outs_occflow=outs_occ,
                    command=command,
                    sdc_planning=copy.deepcopy(sdc_planning),
                    sdc_planning_mask=copy.deepcopy(sdc_planning_mask),
                    gt_future_boxes=copy.deepcopy(gt_future_boxes),
                )
                losses_planning = outs_planning["losses"]
                losses_planning = self.loss_weighted_and_prefixed(
                    losses_planning, prefix="planning"
                )
                losses.update(losses_planning)
                # print(losses_planning[list(losses_planning.keys())[0]].grad_fn)
                planning_gt = dict(
                    segmentation=gt_segmentation,
                    sdc_planning=sdc_planning,
                    sdc_planning_mask=sdc_planning_mask,
                    command=command,
                )
                result[0]["planning"] = dict(
                    planning_gt=planning_gt,
                    result_planning=result_planning,
                )

            for k, v in losses.items():
                losses[k] = torch.nan_to_num(v)
            adv_loss, adv_log_vars = self._parse_losses(losses)
            plan_loss = self.get_task_loss(losses_planning, 'planning')
            # print(losses_planning)
            if (
                self.pgd_cfg.attack_img
                or self.pgd_cfg.attack_track_motion
                or self.pgd_cfg.attack_seg_motion
                or self.pgd_cfg.attack_motion_occ
                or self.pgd_cfg.attack_motion_plan
            ):
                noises = [
                    noise_img,
                    noise_track,
                    noise_seg,
                    noise_motion_traj,
                    noise_motion_sdc_traj,
                ]
                noises_require_grad = list(filter(lambda n: n.requires_grad, noises))
                # print(len(noises_require_grad))
                if plan_loss.grad_fn == None:
                    continue

                grads = torch.autograd.grad(
                    outputs=plan_loss,  # adv_loss
                    inputs=noises_require_grad,
                    allow_unused=True,  
                ) 
            index = 0
            if self.pgd_cfg.attack_img:
                noise_img_grad = grads[index]
                index = index + 1
                assert noise_img_grad is not None
                noise_img = noise_img + self.pgd_cfg.img_alpha * noise_img_grad.sign()
                del noise_img_grad
            if self.pgd_cfg.attack_track_motion:
                noise_track_grad = grads[index]
                index = index + 1
                assert noise_track_grad is not None
                noise_track = (
                    noise_track + self.pgd_cfg.track_alpha * noise_track_grad.sign()
                )
                del noise_track_grad
            if self.pgd_cfg.attack_seg_motion:
                noise_seg_grad = grads[index]
                index = index + 1
                assert noise_seg_grad is not None
                noise_seg = noise_seg + self.pgd_cfg.seg_alpha * noise_seg_grad.sign()
                del noise_seg_grad
            if self.pgd_cfg.attack_motion_occ:
                noise_motion_traj_grad = grads[index]
                index = index + 1
                assert noise_motion_traj_grad is not None
                noise_motion_traj = (
                    noise_motion_traj
                    + self.pgd_cfg.motion_traj_alpha * noise_motion_traj_grad.sign()
                )
                del noise_motion_traj_grad
            if self.pgd_cfg.attack_motion_plan:
                noise_motion_sdc_traj_grad = grads[index]
                index = index + 1
                assert noise_motion_sdc_traj_grad is not None
                noise_motion_sdc_traj = (
                    noise_motion_sdc_traj
                    + self.pgd_cfg.motion_traj_alpha * noise_motion_sdc_traj_grad.sign()
                )
                
        del tmp_sdc_traj_query, tmp_track_query, tmp_traj_query
        torch.cuda.empty_cache()
        result_track = [outs_track]
        pop_track_list = [
            "prev_bev",
            "bev_pos",
            "bev_embed",
            "track_query_embeddings",
            "sdc_embedding",
        ]
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)

        if self.with_seg_head:
            result_seg[0] = pop_elem_in_result(
                result_seg[0], pop_list=["args_tuple"]
            )  # 'pts_bbox',用于vis地
        if self.with_motion_head:
            result_motion[0] = pop_elem_in_result(result_motion[0])
        if self.with_occ_head:
            result[0]["occ"] = pop_elem_in_result(
                result[0]["occ"],
                pop_list=[
                    "seg_out_mask",
                    "flow_out",
                    "future_states_occ",
                    "pred_ins_masks",
                    "pred_raw_occ",
                    "pred_ins_logits",
                    "pred_ins_sigmoid",
                ],
            )

        for i, res in enumerate(result):
            # res['token'] = img_metas[i][len_queue-1]['sample_idx']
            res["token"] = img_metas[i]["sample_idx"]
            res.update(result_track[i])
            if self.with_motion_head:
                res.update(result_motion[i])
            if self.with_seg_head:
                res.update(result_seg[i])
        for i, res in enumerate(result):
            for k, v in res.items():
                if isinstance(v, torch.Tensor):
                    res[k] = v.detach()  # .cpu()
                elif isinstance(v, dict):
                    for dk, dv in v.items():
                        if isinstance(dv, torch.Tensor):
                            res[k][dk] = dv.detach()  # .cpu()
                        elif isinstance(dv, dict):
                            for ddk, ddv in dv.items():
                                if isinstance(ddv, torch.Tensor):
                                    res[k][dk][ddk] = ddv.detach()  # .cpu()
                        #         else:
                        #             print('dict for4', type(ddv))
                        # else:
                        #     print('dict for3', type(dv))
                elif isinstance(v, list):
                    for li, lv in enumerate(v):
                        if isinstance(lv, list):
                            for lli, llv in enumerate(lv):
                                if isinstance(llv, torch.Tensor):
                                    res[k][li][lli] = llv.detach()
                                elif isinstance(llv, LiDARInstance3DBoxes):
                                    llv.tensor = llv.tensor.detach()
                        # else:
                        #     print('list for3', type(dv))
                elif isinstance(v, LiDARInstance3DBoxes):
                    v.tensor = v.tensor.detach()

        return result

    def forward_attack_all_loss_res_imgok(
        self,
        img=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_inds=None,
        l2g_t=None,
        l2g_r_mat=None,
        timestamp=None,
        gt_lane_labels=None,
        gt_lane_bboxes=None,
        gt_lane_masks=None,
        rescale=False,
        gt_fut_traj=None,
        gt_fut_traj_mask=None,
        gt_past_traj=None,
        gt_past_traj_mask=None,
        gt_sdc_bbox=None,
        gt_sdc_label=None,
        gt_sdc_fut_traj=None,
        gt_sdc_fut_traj_mask=None,
        # Occ_gt
        gt_segmentation=None,
        gt_instance=None,
        gt_occ_img_is_valid=None,
        # planning
        sdc_planning=None,
        sdc_planning_mask=None,
        command=None,
        # fut gt for planning
        gt_future_boxes=None,
        **kwargs,  # [1, 9]
    ):
        meta_data = img_metas[0][0]

        # 提取 filename
        file_names = meta_data["filename"]

        # 打印 filename
        for param in self.parameters():
            param.requires_grad = False
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            # the first sample of each scene is truncated
            self.prev_frame_info["prev_bev"] = None
        # update idx
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])
        # first frame
        if self.prev_frame_info["scene_token"] is None:
            img_metas[0][0]["can_bus"][:3] = 0
            img_metas[0][0]["can_bus"][-1] = 0
        # following frames
        else:
            img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle

        if isinstance(img, list):
            img = img[0]
        if img.dim() != 5:
            # bs, nq, n, C, H, W -> bs, n, C, H, W
            img = torch.squeeze(img, dim=1)
        if isinstance(img_metas[0], list):  # test
            img_metas = img_metas[0]
        elif isinstance(img_metas[0], dict):  # train
            img_metas[0] = img_metas[0][0]
        timestamp = timestamp[0]
        if isinstance(timestamp, list):
            timestamp = timestamp[0] if timestamp is not None else None
        result = [dict()]

        if (
            self.pgd_cfg.attack_img
            or self.pgd_cfg.attack_track_motion
            or self.pgd_cfg.attack_seg_motion
            or self.pgd_cfg.attack_motion_occ
            or self.pgd_cfg.attack_motion_plan
        ):
            lun = self.pgd_cfg.steps + 1
        else:
            lun = 1

        for pgd_i in range(lun):
            losses = dict()

            if pgd_i == 0:
                if self.pgd_cfg.attack_img and self.pgd_cfg.random_start:
                    noise_img = torch.empty_like(img).uniform_(
                        -self.pgd_cfg.img_eps, self.pgd_cfg.img_eps
                    )
                else:
                    noise_img = torch.zeros_like(img)
                if self.pgd_cfg.attack_img:
                    noise_img.requires_grad_(True)
            adv_img = img
            if self.pgd_cfg.attack_img:
                adv_img = img + noise_img

            track_img_metas = copy.deepcopy(img_metas)  # !!!!!!!!!!!!
            losses_track, outs_track = self.simple_attack_track(
                adv_img,
                gt_bboxes_3d,
                gt_labels_3d,
                gt_past_traj,
                gt_past_traj_mask,
                gt_inds,
                gt_sdc_bbox,
                gt_sdc_label,
                l2g_t[0][0].unsqueeze(0),
                l2g_r_mat[0][0].unsqueeze(0),
                track_img_metas,
                timestamp,
            )
            losses_track = self.loss_weighted_and_prefixed(losses_track, prefix="track")
            losses.update(losses_track)
            # print(losses_track[list(losses_track.keys())[0]])

            # Upsample bev for tiny version
            outs_track = self.upsample_bev_if_tiny(
                outs_track
            )
            result_track = [{} for _ in range(img.shape[0])]
            result_track[0] = outs_track
            bev_embed = outs_track["bev_embed"]
            bev_pos = outs_track["bev_pos"]

            seg_img_metas = copy.deepcopy(track_img_metas)  
            outs_seg = dict()
            if self.with_seg_head:
                losses_seg, outs_seg, result_seg = self.seg_head.forward_attack(
                    bev_embed,
                    seg_img_metas,
                    gt_lane_labels,
                    gt_lane_bboxes,
                    gt_lane_masks,
                )
                losses_seg = self.loss_weighted_and_prefixed(losses_seg, prefix="map")
                losses.update(losses_seg)
            if 0 == pgd_i:
                assert (
                    outs_track["track_query_embeddings"].shape[-1]
                    == outs_track["sdc_embedding"].shape[-1]
                )
                cat_outs_track = torch.cat(
                    [
                        outs_track["sdc_embedding"][None, :],
                        outs_track["sdc_embedding"][None, :],
                    ],
                    dim=0,
                )
                # print(cat_outs.shape) [2, 256]
                if self.pgd_cfg.attack_track_motion:
                    if self.pgd_cfg.random_start:
                        noise_track = torch.empty_like(cat_outs_track).uniform_(
                            -self.pgd_cfg.track_eps, self.pgd_cfg.track_eps
                        )
                    else:
                        noise_track = torch.zeros_like(cat_outs_track)
                    noise_track.requires_grad_(True)
                else:
                    noise_track = torch.zeros_like(cat_outs_track)

            adv_outs_track = outs_track
            if self.pgd_cfg.attack_track_motion:
                if adv_outs_track["track_query_embeddings"].shape[0] != 0:
                    adv_outs_track["track_query_embeddings"] = adv_outs_track[
                        "track_query_embeddings"
                    ] + torch.cat(
                        [
                            noise_track[0, :][None, :]
                            for _ in range(
                                adv_outs_track["track_query_embeddings"].shape[0]
                            )
                        ],
                        dim=0,
                    )  # noise_track[0:-1, :]
                adv_outs_track["sdc_embedding"] = (
                    adv_outs_track["sdc_embedding"] + noise_track[-1, :]
                )

            if 0 == pgd_i:
                assert (
                    outs_seg["args_tuple"][3].shape == outs_seg["args_tuple"][5].shape
                )
                cat_outs_seg = torch.cat(
                    [outs_seg["args_tuple"][3][None], outs_seg["args_tuple"][5][None]],
                    dim=0,
                )
                if self.pgd_cfg.attack_seg_motion:
                    if self.pgd_cfg.random_start:
                        noise_seg = torch.empty_like(cat_outs_seg).uniform_(
                            -self.pgd_cfg.seg_eps, self.pgd_cfg.seg_eps
                        )
                    else:
                        noise_seg = torch.zeros_like(cat_outs_seg)
                    noise_seg.requires_grad_(True)
                else:
                    noise_seg = torch.zeros_like(cat_outs_seg)

            adv_outs_seg = outs_seg
            if self.pgd_cfg.attack_seg_motion:
                adv_outs_seg["args_tuple"][3] = (
                    adv_outs_seg["args_tuple"][3] + noise_seg[0]
                )
                adv_outs_seg["args_tuple"][5] = (
                    adv_outs_seg["args_tuple"][5] + noise_seg[1]
                )

            outs_motion = dict()
            # Forward Motion Head
            if self.with_motion_head:
                ret_dict_motion, result_motion = self.motion_head.forward_attack(
                    bev_embed,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    gt_fut_traj,
                    gt_fut_traj_mask,
                    gt_sdc_fut_traj,
                    gt_sdc_fut_traj_mask,
                    outs_track=adv_outs_track,
                    outs_seg=adv_outs_seg,
                )
                losses_motion = ret_dict_motion["losses"]
                outs_motion = ret_dict_motion["outs_motion"]
                outs_motion["bev_pos"] = bev_pos
                losses_motion = self.loss_weighted_and_prefixed(
                    losses_motion, prefix="motion"
                )
                losses.update(losses_motion)

            if self.with_occ_head:
                occ_no_query = outs_motion["track_query"].shape[1] == 0
                if outs_motion["track_query"].shape[1] == 0: 
                    # TODO: rm hard code
                    outs_motion["track_query"] = torch.zeros((1, 1, 256)).to(bev_embed)
                    outs_motion["track_query_pos"] = torch.zeros((1, 1, 256)).to(
                        bev_embed
                    )
                    outs_motion["traj_query"] = torch.zeros((3, 1, 1, 6, 256)).to(
                        bev_embed
                    )
                    outs_motion["all_matched_idxes"] = [[-1]]
            if 0 == pgd_i:
                assert "traj_query" in outs_motion.keys()
                assert (
                    outs_motion["track_query"].shape
                    == outs_motion["track_query_pos"].shape
                )
                assert (
                    outs_motion["track_query"].shape[0]
                    == outs_motion["traj_query"].shape[1]
                )
                assert (
                    outs_motion["track_query"].shape[1]
                    == outs_motion["traj_query"].shape[2]
                )
                assert (
                    outs_motion["track_query"].shape[-1]
                    == outs_motion["traj_query"].shape[-1]
                )
                tmp_traj_query = outs_motion["traj_query"][-1]
                tmp_traj_query = tmp_traj_query[:, 0:1, :, :] 
                tmp_track_query = outs_motion["track_query"][None, :, 0:1, :]
                if self.pgd_cfg.attack_motion_occ:
                    if self.pgd_cfg.random_start:
                        noise_motion_traj = torch.empty_like(tmp_traj_query).uniform_(
                            -self.pgd_cfg.motion_traj_eps, self.pgd_cfg.motion_traj_eps
                        )
                    else:
                        noise_motion_traj = torch.zeros_like(tmp_traj_query)
                    noise_motion_traj.requires_grad_(True)
                else:
                    noise_motion_traj = torch.zeros_like(tmp_traj_query)

            adv_outs_motion = outs_motion
            if self.pgd_cfg.attack_motion_occ:
                assert "traj_query" in adv_outs_motion.keys()
                # 1, 36, 6, 256
                if outs_motion["track_query"].shape[1] != 0:
                    adv_outs_motion["traj_query"][-1] = adv_outs_motion["traj_query"][
                        -1
                    ] + torch.cat(
                        [
                            noise_motion_traj
                            for _ in range(adv_outs_motion["traj_query"][-1].shape[1])
                        ],
                        dim=1,
                    )  
                else:
                    print("motion occ   00000")
                
            # Forward Occ Head
            if self.with_occ_head:
                losses_occ, outs_occ = self.occ_head.forward_attack(
                    bev_feat=bev_embed,
                    outs_dict=adv_outs_motion,
                    no_query=occ_no_query,
                    gt_inds_list=copy.deepcopy(gt_inds),
                    gt_segmentation=copy.deepcopy(gt_segmentation),
                    gt_instance=copy.deepcopy(gt_instance),
                    gt_img_is_valid=copy.deepcopy(gt_occ_img_is_valid),
                )
                losses_occ = self.loss_weighted_and_prefixed(losses_occ, prefix="occ")
                losses.update(losses_occ)
                result[0]["occ"] = outs_occ

            adv_outs_motion = outs_motion
            if 0 == pgd_i:
                # 只用了-1
                tmp_sdc_traj_query = adv_outs_motion["sdc_traj_query"][-1]  # 1,6,256
                if self.pgd_cfg.attack_motion_plan:
                    if self.pgd_cfg.random_start:
                        noise_motion_sdc_traj = torch.empty_like(
                            tmp_sdc_traj_query
                        ).uniform_(
                            -self.pgd_cfg.motion_traj_eps, self.pgd_cfg.motion_traj_eps
                        )
                    else:
                        noise_motion_sdc_traj = torch.zeros_like(tmp_sdc_traj_query)
                    noise_motion_sdc_traj.requires_grad_(True)
                else:
                    noise_motion_sdc_traj = torch.zeros_like(tmp_sdc_traj_query)
            if self.pgd_cfg.attack_motion_plan:
                # 1, 6, 256
                adv_outs_motion["sdc_traj_query"][-1] = (
                    adv_outs_motion["sdc_traj_query"][-1] + noise_motion_sdc_traj
                )

            # Forward Plan Head
            if self.with_planning_head:
                outs_planning, result_planning = self.planning_head.forward_attack(
                    bev_embed=bev_embed,
                    outs_motion=adv_outs_motion,
                    outs_occflow=outs_occ,
                    command=command,
                    sdc_planning=copy.deepcopy(sdc_planning),
                    sdc_planning_mask=copy.deepcopy(sdc_planning_mask),
                    gt_future_boxes=copy.deepcopy(gt_future_boxes),
                )
                losses_planning = outs_planning["losses"]
                losses_planning = self.loss_weighted_and_prefixed(
                    losses_planning, prefix="planning"
                )
                losses.update(losses_planning)
                planning_gt = dict(
                    segmentation=gt_segmentation,
                    sdc_planning=sdc_planning,
                    sdc_planning_mask=sdc_planning_mask,
                    command=command,
                )
                result[0]["planning"] = dict(
                    planning_gt=planning_gt,
                    result_planning=result_planning,
                )

            for k, v in losses.items():
                losses[k] = torch.nan_to_num(v)
            adv_loss, adv_log_vars = self._parse_losses(losses)
            if (
                self.pgd_cfg.attack_img
                or self.pgd_cfg.attack_track_motion
                or self.pgd_cfg.attack_seg_motion
                or self.pgd_cfg.attack_motion_occ
                or self.pgd_cfg.attack_motion_plan
            ):
                noises = [
                    noise_img,
                    noise_track,
                    noise_seg,
                    noise_motion_traj,
                    noise_motion_sdc_traj,
                ]
                noises_require_grad = list(filter(lambda n: n.requires_grad, noises))
                if adv_loss.grad_fn == None:
                    continue

                grads = torch.autograd.grad(
                    outputs=adv_loss,  # adv_loss
                    inputs=noises_require_grad,
                    allow_unused=True,
                ) 
            index = 0
            if self.pgd_cfg.attack_img:
                noise_img_grad = grads[index]
                index = index + 1
                assert noise_img_grad is not None
                noise_img = noise_img + self.pgd_cfg.img_alpha * noise_img_grad.sign()
                del noise_img_grad
            if self.pgd_cfg.attack_track_motion:
                noise_track_grad = grads[index]
                index = index + 1
                assert noise_track_grad is not None
                noise_track = (
                    noise_track + self.pgd_cfg.track_alpha * noise_track_grad.sign()
                )
                del noise_track_grad
            if self.pgd_cfg.attack_seg_motion:
                noise_seg_grad = grads[index]
                index = index + 1
                assert noise_seg_grad is not None
                noise_seg = noise_seg + self.pgd_cfg.seg_alpha * noise_seg_grad.sign()
                del noise_seg_grad
            if self.pgd_cfg.attack_motion_occ:
                noise_motion_traj_grad = grads[index]
                index = index + 1
                assert noise_motion_traj_grad is not None
                noise_motion_traj = (
                    noise_motion_traj
                    + self.pgd_cfg.motion_traj_alpha * noise_motion_traj_grad.sign()
                )
                del noise_motion_traj_grad
            if self.pgd_cfg.attack_motion_plan:
                noise_motion_sdc_traj_grad = grads[index]
                index = index + 1
                assert noise_motion_sdc_traj_grad is not None
                noise_motion_sdc_traj = (
                    noise_motion_sdc_traj
                    + self.pgd_cfg.motion_traj_alpha * noise_motion_sdc_traj_grad.sign()
                )
                
        del tmp_sdc_traj_query, tmp_track_query, tmp_traj_query
        torch.cuda.empty_cache()
        result_track = [outs_track]
        pop_track_list = [
            "prev_bev",
            "bev_pos",
            "bev_embed",
            "track_query_embeddings",
            "sdc_embedding",
        ]
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)

        if self.with_seg_head:
            result_seg[0] = pop_elem_in_result(
                result_seg[0], pop_list=["args_tuple"]
            )  # 'pts_bbox',用于vis地
        if self.with_motion_head:
            result_motion[0] = pop_elem_in_result(result_motion[0])
        if self.with_occ_head:
            result[0]["occ"] = pop_elem_in_result(
                result[0]["occ"],
                pop_list=[
                    "seg_out_mask",
                    "flow_out",
                    "future_states_occ",
                    "pred_ins_masks",
                    "pred_raw_occ",
                    "pred_ins_logits",
                    "pred_ins_sigmoid",
                ],
            )

        for i, res in enumerate(result):
            # res['token'] = img_metas[i][len_queue-1]['sample_idx']
            res["token"] = img_metas[i]["sample_idx"]
            res.update(result_track[i])
            if self.with_motion_head:
                res.update(result_motion[i])
            if self.with_seg_head:
                res.update(result_seg[i])
        for i, res in enumerate(result):
            for k, v in res.items():
                if isinstance(v, torch.Tensor):
                    res[k] = v.detach()  # .cpu()
                elif isinstance(v, dict):
                    for dk, dv in v.items():
                        if isinstance(dv, torch.Tensor):
                            res[k][dk] = dv.detach()  # .cpu()
                        elif isinstance(dv, dict):
                            for ddk, ddv in dv.items():
                                if isinstance(ddv, torch.Tensor):
                                    res[k][dk][ddk] = ddv.detach()  # .cpu()
                        #         else:
                        #             print('dict for4', type(ddv))
                        # else:
                        #     print('dict for3', type(dv))
                elif isinstance(v, list):
                    for li, lv in enumerate(v):
                        if isinstance(lv, list):
                            for lli, llv in enumerate(lv):
                                if isinstance(llv, torch.Tensor):
                                    res[k][li][lli] = llv.detach()
                                elif isinstance(llv, LiDARInstance3DBoxes):
                                    llv.tensor = llv.tensor.detach()
                        # else:
                        #     print('list for3', type(dv))
                elif isinstance(v, LiDARInstance3DBoxes):
                    v.tensor = v.tensor.detach()

        return result

    # Add the subtask loss to the whole model loss
    @auto_fp16(apply_to=("img", "points"))
    def forward_train(
        self,
        img=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_inds=None,
        l2g_t=None,
        l2g_r_mat=None,
        timestamp=None,
        gt_lane_labels=None,
        gt_lane_bboxes=None,
        gt_lane_masks=None,
        gt_fut_traj=None,
        gt_fut_traj_mask=None,
        gt_past_traj=None,
        gt_past_traj_mask=None,
        gt_sdc_bbox=None,
        gt_sdc_label=None,
        gt_sdc_fut_traj=None,
        gt_sdc_fut_traj_mask=None,
        # Occ_gt
        gt_segmentation=None,
        gt_instance=None,
        gt_occ_img_is_valid=None,
        # planning
        sdc_planning=None,
        sdc_planning_mask=None,
        command=None,
        # fut gt for planning
        gt_future_boxes=None,
        **kwargs,  # [1, 9]
    ):
        """Forward training function for the model that includes multiple tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning.

        Args:
        img (torch.Tensor, optional): Tensor containing images of each sample with shape (N, C, H, W). Defaults to None.
        img_metas (list[dict], optional): List of dictionaries containing meta information for each sample. Defaults to None.
        gt_bboxes_3d (list[:obj:BaseInstance3DBoxes], optional): List of ground truth 3D bounding boxes for each sample. Defaults to None.
        gt_labels_3d (list[torch.Tensor], optional): List of tensors containing ground truth labels for 3D bounding boxes. Defaults to None.
        gt_inds (list[torch.Tensor], optional): List of tensors containing indices of ground truth objects. Defaults to None.
        l2g_t (list[torch.Tensor], optional): List of tensors containing translation vectors from local to global coordinates. Defaults to None.
        l2g_r_mat (list[torch.Tensor], optional): List of tensors containing rotation matrices from local to global coordinates. Defaults to None.
        timestamp (list[float], optional): List of timestamps for each sample. Defaults to None.
        gt_bboxes_ignore (list[torch.Tensor], optional): List of tensors containing ground truth 2D bounding boxes in images to be ignored. Defaults to None.
        gt_lane_labels (list[torch.Tensor], optional): List of tensors containing ground truth lane labels. Defaults to None.
        gt_lane_bboxes (list[torch.Tensor], optional): List of tensors containing ground truth lane bounding boxes. Defaults to None.
        gt_lane_masks (list[torch.Tensor], optional): List of tensors containing ground truth lane masks. Defaults to None.
        gt_fut_traj (list[torch.Tensor], optional): List of tensors containing ground truth future trajectories. Defaults to None.
        gt_fut_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth future trajectory masks. Defaults to None.
        gt_past_traj (list[torch.Tensor], optional): List of tensors containing ground truth past trajectories. Defaults to None.
        gt_past_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth past trajectory masks. Defaults to None.
        gt_sdc_bbox (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car bounding boxes. Defaults to None.
        gt_sdc_label (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car labels. Defaults to None.
        gt_sdc_fut_traj (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car future trajectories. Defaults to None.
        gt_sdc_fut_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car future trajectory masks. Defaults to None.
        gt_segmentation (list[torch.Tensor], optional): List of tensors containing ground truth segmentation masks. Defaults to
        gt_instance (list[torch.Tensor], optional): List of tensors containing ground truth instance segmentation masks. Defaults to None.
        gt_occ_img_is_valid (list[torch.Tensor], optional): List of tensors containing binary flags indicating whether an image is valid for occupancy prediction. Defaults to None.
        sdc_planning (list[torch.Tensor], optional): List of tensors containing self-driving car planning information. Defaults to None.
        sdc_planning_mask (list[torch.Tensor], optional): List of tensors containing self-driving car planning masks. Defaults to None.
        command (list[torch.Tensor], optional): List of tensors containing high-level command information for planning. Defaults to None.
        gt_future_boxes (list[torch.Tensor], optional): List of tensors containing ground truth future bounding boxes for planning. Defaults to None.
        gt_future_labels (list[torch.Tensor], optional): List of tensors containing ground truth future labels for planning. Defaults to None.

        Returns:
            dict: Dictionary containing losses of different tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning. Each key in the dictionary
                is prefixed with the corresponding task name, e.g., 'track', 'map', 'motion', 'occ', and 'planning'. The values are the calculated losses for each task.
        """
        losses = dict()
        # print(type(img))    # <class 'torch.Tensor'>
        len_queue = img.size(1)

        track_img_metas = copy.deepcopy(img_metas)  # !!!!!!!!!!!!
        losses_track, outs_track = self.forward_track_train(
            img,
            gt_bboxes_3d,
            gt_labels_3d,
            gt_past_traj,
            gt_past_traj_mask,
            gt_inds,
            gt_sdc_bbox,
            gt_sdc_label,
            l2g_t,
            l2g_r_mat,
            track_img_metas,
            timestamp,
        )
        losses_track = self.loss_weighted_and_prefixed(losses_track, prefix="track")
        losses.update(losses_track)
        # print(losses_track[list(losses_track.keys())[0]])
        # quit()
        # Upsample bev for tiny version
        outs_track = self.upsample_bev_if_tiny(outs_track)
        # print(outs_track['track_query_embeddings'].shape)
        bev_embed = outs_track["bev_embed"]
        bev_pos = outs_track["bev_pos"]

        seg_img_metas = copy.deepcopy(track_img_metas)  # !!!!!!!!!!!!
        seg_img_metas = [each[len_queue - 1] for each in seg_img_metas]

        outs_seg = dict()
        if self.with_seg_head:
            losses_seg, outs_seg = self.seg_head.forward_train(
                bev_embed, seg_img_metas, gt_lane_labels, gt_lane_bboxes, gt_lane_masks
            )

            losses_seg = self.loss_weighted_and_prefixed(losses_seg, prefix="map")
            losses.update(losses_seg)
        # print(outs_seg['args_tuple'][3].shape)      # torch.Size([1, 300, 256])
        # print(outs_seg['args_tuple'][5].shape)      # torch.Size([1, 300, 256])

        outs_motion = dict()
        # Forward Motion Head
        if self.with_motion_head:
            ret_dict_motion = self.motion_head.forward_train(
                bev_embed,
                gt_bboxes_3d,
                gt_labels_3d,
                gt_fut_traj,
                gt_fut_traj_mask,
                gt_sdc_fut_traj,
                gt_sdc_fut_traj_mask,
                outs_track=outs_track,
                outs_seg=outs_seg,
            )
            losses_motion = ret_dict_motion["losses"]
            outs_motion = ret_dict_motion["outs_motion"]
            outs_motion["bev_pos"] = bev_pos
            losses_motion = self.loss_weighted_and_prefixed(
                losses_motion, prefix="motion"
            )
            losses.update(losses_motion)

        # Forward Occ Head
        if self.with_occ_head:
            if outs_motion["track_query"].shape[1] == 0:
                # TODO: rm hard code
                outs_motion["track_query"] = torch.zeros((1, 1, 256)).to(bev_embed)
                outs_motion["track_query_pos"] = torch.zeros((1, 1, 256)).to(bev_embed)
                outs_motion["traj_query"] = torch.zeros((3, 1, 1, 6, 256)).to(bev_embed)
                outs_motion["all_matched_idxes"] = [[-1]]
            losses_occ = self.occ_head.forward_train(
                bev_embed,
                outs_motion,
                gt_inds_list=gt_inds,
                gt_segmentation=gt_segmentation,
                gt_instance=gt_instance,
                gt_img_is_valid=gt_occ_img_is_valid,
            )
            losses_occ = self.loss_weighted_and_prefixed(losses_occ, prefix="occ")
            losses.update(losses_occ)

        # Forward Plan Head
        if self.with_planning_head:
            outs_planning = self.planning_head.forward_train(
                bev_embed,
                outs_motion,
                sdc_planning,
                sdc_planning_mask,
                command,
                gt_future_boxes,
            )
            losses_planning = outs_planning["losses"]
            losses_planning = self.loss_weighted_and_prefixed(
                losses_planning, prefix="planning"
            )
            losses.update(losses_planning)
        # print(self.get_task_loss(losses_planning, 'planning'))
        for k, v in losses.items():
            losses[k] = torch.nan_to_num(v)
        # print(f"'loss' is in return losses: {'loss' in losses.keys()}") False
        # # tensor(251.6305, device='cuda:0', dtype=torch.float64, grad_fn=<AddBackward0>)
        return losses

    def loss_weighted_and_prefixed(self, loss_dict, prefix=""):
        loss_factor = self.task_loss_weight[prefix]
        loss_dict = {f"{prefix}.{k}": v * loss_factor for k, v in loss_dict.items()}
        return loss_dict

    def forward_test(
        self,
        img=None,
        img_metas=None,
        l2g_t=None,
        l2g_r_mat=None,
        timestamp=None,
        gt_lane_labels=None,
        gt_lane_masks=None,
        rescale=False,
        # planning gt(for evaluation only)
        sdc_planning=None,
        sdc_planning_mask=None,
        command=None,
        # Occ_gt (for evaluation only)
        gt_segmentation=None,
        gt_instance=None,
        gt_occ_img_is_valid=None,
        **kwargs,
    ):
        """Test function"""
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            # the first sample of each scene is truncated
            self.prev_frame_info["prev_bev"] = None
        # update idx
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])
        # first frame
        if self.prev_frame_info["scene_token"] is None:
            img_metas[0][0]["can_bus"][:3] = 0
            img_metas[0][0]["can_bus"][-1] = 0
        # following frames
        else:
            img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle

        img = img[0]
        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None
        result = [dict() for i in range(len(img_metas))]
        result_track = self.simple_test_track(
            img, l2g_t, l2g_r_mat, img_metas, timestamp
        )

        # Upsample bev for tiny model
        result_track[0] = self.upsample_bev_if_tiny(result_track[0])

        bev_embed = result_track[0]["bev_embed"]

        if self.with_seg_head:
            result_seg = self.seg_head.forward_test(
                bev_embed, gt_lane_labels, gt_lane_masks, img_metas, rescale
            )

        if self.with_motion_head:
            result_motion, outs_motion = self.motion_head.forward_test(
                bev_embed, outs_track=result_track[0], outs_seg=result_seg[0]
            )
            outs_motion["bev_pos"] = result_track[0]["bev_pos"]

        outs_occ = dict()
        if self.with_occ_head:
            occ_no_query = outs_motion["track_query"].shape[1] == 0
            outs_occ = self.occ_head.forward_test(
                bev_embed,
                outs_motion,
                no_query=occ_no_query,
                gt_segmentation=gt_segmentation,
                gt_instance=gt_instance,
                gt_img_is_valid=gt_occ_img_is_valid,
            )
            result[0]["occ"] = outs_occ

        if self.with_planning_head:
            planning_gt = dict(
                segmentation=gt_segmentation,
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                command=command,
            )
            result_planning = self.planning_head.forward_test(
                bev_embed, outs_motion, outs_occ, command
            )
            result[0]["planning"] = dict(
                planning_gt=planning_gt,
                result_planning=result_planning,
            )

        pop_track_list = [
            "prev_bev",
            "bev_pos",
            "bev_embed",
            "track_query_embeddings",
            "sdc_embedding",
        ]
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)
        # quit()
        if self.with_seg_head:
            # for k, v in result_seg[0]['pts_bbox'].items():
            #     if isinstance(v, torch.Tensor):
            #         result_seg[0]['pts_bbox'][k] = v.detach().cpu()
            result_seg[0] = pop_elem_in_result(
                result_seg[0], pop_list=["args_tuple", "pts_bbox"]
            )  # 'pts_bbox',用于vis地图
        if self.with_motion_head:
            result_motion[0] = pop_elem_in_result(result_motion[0])
        if self.with_occ_head:
            result[0]["occ"] = pop_elem_in_result(
                result[0]["occ"],
                pop_list=[
                    "seg_out_mask",
                    "flow_out",
                    "future_states_occ",
                    "pred_ins_masks",
                    "pred_raw_occ",
                    "pred_ins_logits",
                    "pred_ins_sigmoid",
                ],
            )

        for i, res in enumerate(result):
            res["token"] = img_metas[i]["sample_idx"]
            res.update(result_track[i])
            if self.with_motion_head:
                res.update(result_motion[i])
            if self.with_seg_head:
                res.update(result_seg[i])

        return result


def pop_elem_in_result(task_result: dict, pop_list: list = None):
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith("query") or k.endswith("query_pos") or k.endswith("embedding"):
            task_result.pop(k)

    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result
