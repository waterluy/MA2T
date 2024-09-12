import time
import copy

import torch
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmcv.runner import force_fp32, auto_fp16
from scipy.optimize import linear_sum_assignment
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.VAD.planner.metric_stp3 import PlanningMetric

from projects.mmdet3d_plugin.VAD.adv import ADV
import mmcv
import os

import imgaug.augmenters as iaa
import imgaug.augmenters.imgcorruptlike as icl
import numpy as np


@DETECTORS.register_module()
class VAD(MVXTwoStageDetector):
    """VAD model.
    """
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 fut_ts=6,
                 fut_mode=6
                 ):

        super(VAD,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.valid_fut_ts = pts_bbox_head['valid_fut_ts']

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        self.planning_metric = None
        
        self.attack = False
        self.save_noise = False
        self.black = False
        self.transfer = None
        self.natural = None
        self.dynamic = None
        self.n_tasks = 4
        self.ma2t_task_weights = np.ones(self.n_tasks, dtype=np.float32)
        self.task_dict = {"detection": 0, "map": 1, "motion": 2, "plan": 3}
        self.target = 'all'

    def set_at(self, value):
        self.adv_train = value
        self.pts_bbox_head.adv_train = value

    def set_attack(self, value):
        self.attack = value
        self.pts_bbox_head.attack = value

    @property
    def adv(self) -> ADV:
        return self.pts_bbox_head.adv
    
    @adv.setter
    def adv(self, value):
        self.pts_bbox_head.adv = value

    @property
    def attack_img(self):
        return self.pts_bbox_head.adv.attack_img
    
    @property
    def noise_img(self):
        return self.pts_bbox_head.adv.noise_img
    
    @noise_img.setter
    def noise_img(self, value):
        self.pts_bbox_head.adv.noise_img = value

    @property
    def attack_map(self):
        return self.pts_bbox_head.adv.attack_map
    
    @property
    def noise_map(self):
        return self.pts_bbox_head.adv.noise_map
    
    @noise_img.setter
    def noise_map(self, value):
        self.pts_bbox_head.adv.noise_map = value

    @property
    def attack_agent(self):
        return self.pts_bbox_head.adv.attack_agent
    
    @property
    def noise_agent(self):
        return self.pts_bbox_head.adv.noise_agent
    
    @noise_img.setter
    def noise_agent(self, value):
        self.pts_bbox_head.adv.noise_agent = value
    
    @property
    def steps(self):
        return 1
        # return self.pts_bbox_head.adv.steps

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          map_gt_bboxes_3d,
                          map_gt_labels_3d,                          
                          img_metas,
                          gt_bboxes_ignore=None,
                          map_gt_bboxes_ignore=None,
                          prev_bev=None,
                          ego_his_trajs=None,
                          ego_fut_trajs=None,
                          ego_fut_masks=None,
                          ego_fut_cmd=None,
                          ego_lcf_feat=None,
                          gt_attr_labels=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev,
                                  ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat)
        loss_inputs = [
            gt_bboxes_3d, gt_labels_3d, map_gt_bboxes_3d, map_gt_labels_3d,
            outs, ego_fut_trajs, ego_fut_masks, ego_fut_cmd, gt_attr_labels
        ]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

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
                return self.forward_adv_train(**kwargs)
            else:
                return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev
        
    def get_task_loss(self, losses, task_name, weight=None):
        # for k, v in losses.items():
        #     losses[k] = torch.nan_to_num(v)
        from collections import OrderedDict
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")
        if task_name == 'detection':
            loss = sum(_value for _key, _value in log_vars.items() if ("loss_cls" in _key) or ("loss_bbox" in _key))
        elif task_name == 'map':
            loss = sum(_value for _key, _value in log_vars.items() if ("loss_map" in _key))
        elif task_name == "motion":
            loss = sum(_value for _key, _value in log_vars.items() if ("loss_traj" in _key))
        elif task_name == "plan":
            loss = sum(_value for _key, _value in log_vars.items() if ("loss_plan" in _key))
        if weight is None:
            return {f"{task_name}.loss": loss}
        else:
            return {f"{task_name}.loss": loss * weight}

    # @auto_fp16(apply_to=('img', 'points'))
    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_adv_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      map_gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ego_his_trajs=None,
                      ego_fut_trajs=None,
                      ego_fut_masks=None,
                      ego_fut_cmd=None,
                      ego_lcf_feat=None,
                      gt_attr_labels=None
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue > 1 else None

        ori_img_metas = [each[len_queue-1] for each in img_metas]

        if self.attack_img:
            self.noise_img = torch.zeros_like(img).cuda().requires_grad_(True)
        if self.attack_map:
            self.noise_map = torch.zeros(torch.Size([1, 100, 20, 256])).cuda().requires_grad_(True)
        if self.attack_agent:
            self.noise_agent = torch.zeros(torch.Size([1, 300, 3072])).cuda().requires_grad_(True)

        for step in range(self.steps + 1):
            adv_img = copy.deepcopy(img)        # !!!!!!!!!!!!!!!
            if self.attack_img:
                adv_img = img + self.noise_img
            adv_img_metas = copy.deepcopy(ori_img_metas)        # !!!!!!!!!!!!!!!
            img_feats = self.extract_feat(img=adv_img, img_metas=adv_img_metas)
            
            losses = dict()
            losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d,
                                            map_gt_bboxes_3d, map_gt_labels_3d, adv_img_metas,
                                            gt_bboxes_ignore, map_gt_bboxes_ignore, prev_bev,
                                            ego_his_trajs=ego_his_trajs, ego_fut_trajs=ego_fut_trajs,
                                            ego_fut_masks=ego_fut_masks, ego_fut_cmd=ego_fut_cmd,
                                            ego_lcf_feat=ego_lcf_feat, gt_attr_labels=gt_attr_labels)
            if self.dynamic is None:
                losses.update(losses_pts)
            else:
                original_loss = {}
                for k, v in losses_pts.items():
                    original_loss[k.replace("loss", "originalLOSS")] = v
                losses.update(original_loss)
                # for saving log
                losses.update(self.get_task_loss(losses_pts, 'detection', weight=self.ma2t_task_weights[self.task_dict["detection"]]))
                losses.update(self.get_task_loss(losses_pts, 'map', weight=self.ma2t_task_weights[self.task_dict["map"]]))
                losses.update(self.get_task_loss(losses_pts, 'motion', weight=self.ma2t_task_weights[self.task_dict["motion"]]))
                losses.update(self.get_task_loss(losses_pts, 'plan', weight=self.ma2t_task_weights[self.task_dict["plan"]]))
            
            adv_loss, adv_log_vars = self._parse_losses(losses)
            if step == self.steps:
                break
            self.adv.update_noise(loss=adv_loss)

        return adv_loss, adv_log_vars

    # @auto_fp16(apply_to=('img', 'points'))
    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      map_gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ego_his_trajs=None,
                      ego_fut_trajs=None,
                      ego_fut_masks=None,
                      ego_fut_cmd=None,
                      ego_lcf_feat=None,
                      gt_attr_labels=None
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        # prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        # import pdb;pdb.set_trace()
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue > 1 else None

        img_metas = [each[len_queue-1] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d,
                                            map_gt_bboxes_3d, map_gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, map_gt_bboxes_ignore, prev_bev,
                                            ego_his_trajs=ego_his_trajs, ego_fut_trajs=ego_fut_trajs,
                                            ego_fut_masks=ego_fut_masks, ego_fut_cmd=ego_fut_cmd,
                                            ego_lcf_feat=ego_lcf_feat, gt_attr_labels=gt_attr_labels)

        losses.update(losses_pts)
        return losses

    def forward_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs
    ):
        # print(kwargs.keys())
        # dict_keys(['rescale', 'points', 'fut_valid_flag', 'ego_fut_masks', 'map_gt_labels_3d', 'map_gt_bboxes_3d'])
        # quit()
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        if self.attack:
            if self.natural is not None:
                new_prev_bev, bbox_results = self.simple_attack_natural(
                    img_metas=img_metas[0],
                    img=img[0],
                    prev_bev=self.prev_frame_info['prev_bev'],
                    gt_bboxes_3d=gt_bboxes_3d,
                    gt_labels_3d=gt_labels_3d,
                    ego_his_trajs=ego_his_trajs[0],
                    ego_fut_trajs=ego_fut_trajs[0],
                    ego_fut_cmd=ego_fut_cmd[0],
                    ego_lcf_feat=ego_lcf_feat[0],
                    gt_attr_labels=gt_attr_labels,
                    **kwargs
                )
            elif self.transfer == "uniad":
                with torch.no_grad():
                    new_prev_bev, bbox_results = self.simple_attack_transfer_from_uniad(
                        img_metas=img_metas[0],
                        img=img[0],
                        prev_bev=self.prev_frame_info['prev_bev'],
                        gt_bboxes_3d=gt_bboxes_3d,
                        gt_labels_3d=gt_labels_3d,
                        ego_his_trajs=ego_his_trajs[0],
                        ego_fut_trajs=ego_fut_trajs[0],
                        ego_fut_cmd=ego_fut_cmd[0],
                        ego_lcf_feat=ego_lcf_feat[0],
                        gt_attr_labels=gt_attr_labels,
                        **kwargs
                    )
            else:
                new_prev_bev, bbox_results = self.simple_attack(
                    img_metas=img_metas[0],
                    img=img[0],
                    prev_bev=self.prev_frame_info['prev_bev'],
                    gt_bboxes_3d=gt_bboxes_3d,
                    gt_labels_3d=gt_labels_3d,
                    ego_his_trajs=ego_his_trajs[0],
                    ego_fut_trajs=ego_fut_trajs[0],
                    ego_fut_cmd=ego_fut_cmd[0],
                    ego_lcf_feat=ego_lcf_feat[0],
                    gt_attr_labels=gt_attr_labels,
                    **kwargs
                )
        else:
            new_prev_bev, bbox_results = self.simple_test(
                img_metas=img_metas[0],
                img=img[0],
                prev_bev=self.prev_frame_info['prev_bev'],
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                ego_his_trajs=ego_his_trajs[0],
                ego_fut_trajs=ego_fut_trajs[0],
                ego_fut_cmd=ego_fut_cmd[0],
                ego_lcf_feat=ego_lcf_feat[0],
                gt_attr_labels=gt_attr_labels,
                **kwargs
            )
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev

        return bbox_results
    
    def simple_attack(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        prev_bev=None,
        points=None,
        fut_valid_flag=None,
        rescale=False,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_masks=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        map_gt_bboxes_3d=None,
        map_gt_labels_3d=None,
        gt_bboxes_ignore=None,
        map_gt_bboxes_ignore=None,
        # **kwargs
    ):
        # print(kwargs.keys())
        # dict_keys(['ego_fut_masks', 'map_gt_labels_3d', 'map_gt_bboxes_3d'])
        # quit()

        if self.attack_img:
            self.noise_img = torch.zeros_like(img).cuda().requires_grad_(True)
        if self.attack_map:
            self.noise_map = torch.zeros(torch.Size([1, 100, 20, 256])).cuda().requires_grad_(True)
        if self.attack_agent:
            self.noise_agent = torch.zeros(torch.Size([1, 300, 3072])).cuda().requires_grad_(True)

        iter = self.steps + 1
        for step in range(iter):
            adv_img = copy.deepcopy(img)        # !!!!!!!!!!!!!!!
            if self.attack_img:
                adv_img = img + self.noise_img
            adv_img_metas = copy.deepcopy(img_metas)        # !!!!!!!!!!!!!!!
            losses = dict()

            """Test function without augmentaiton."""
            img_feats = self.extract_feat(img=adv_img, img_metas=adv_img_metas)
            bbox_list = [dict() for i in range(len(adv_img_metas))]
            new_prev_bev, bbox_pts, metric_dict, losses_pts = self.simple_attack_pts(
                img_feats,
                adv_img_metas,
                gt_bboxes_3d,
                gt_labels_3d,
                prev_bev,
                fut_valid_flag=fut_valid_flag,
                rescale=rescale,
                start=None,
                ego_his_trajs=ego_his_trajs,
                ego_fut_trajs=ego_fut_trajs,
                ego_fut_masks=ego_fut_masks,
                ego_fut_cmd=ego_fut_cmd,
                ego_lcf_feat=ego_lcf_feat,
                gt_attr_labels=gt_attr_labels,
                map_gt_bboxes_3d=map_gt_bboxes_3d,
                map_gt_labels_3d=map_gt_labels_3d,
            )

            if (not self.black) and (self.natural is None):
                if self.target == "all":
                    losses.update(losses_pts)
                    adv_loss, adv_log_vars = self._parse_losses(losses)
                    self.adv.update_noise(loss=adv_loss)
                elif self.target == "plan":
                    losses.update(self.get_task_loss(losses_pts, 'plan',))
                    plan_loss, _ = self._parse_losses(losses)
                    self.adv.update_noise(loss=plan_loss)
                elif self.target == "task":
                    detection_loss, _ = self._parse_losses(self.get_task_loss(losses_pts, 'detection'))
                    map_loss, _ = self._parse_losses(self.get_task_loss(losses_pts, 'map'))
                    motion_loss, _ = self._parse_losses(self.get_task_loss(losses_pts, 'motion'))
                    plan_loss, _ = self._parse_losses(self.get_task_loss(losses_pts, 'plan'))
                    self.adv.update_noise_task(detection_loss=detection_loss, map_loss=map_loss, motion_loss=motion_loss, plan_loss=plan_loss)
                else:
                    raise Exception(f"self.target: {self.target}")
            
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            if isinstance(pts_bbox, torch.Tensor):
                result_dict['pts_bbox'] = pts_bbox.detach()
            else:   
                result_dict['pts_bbox'] = pts_bbox
            result_dict['metric_results'] = metric_dict
        # print(new_prev_bev.requires_grad)   # True

        return new_prev_bev.detach(), bbox_list

    def simple_attack_pts(
        self,
        x,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        prev_bev=None,
        fut_valid_flag=None,
        rescale=False,
        start=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_masks=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        map_gt_bboxes_3d=None,
        map_gt_labels_3d=None,
    ):
        """Test function"""
        mapped_class_names = [
            'car', 'truck', 'construction_vehicle', 'bus',
            'trailer', 'barrier', 'motorcycle', 'bicycle', 
            'pedestrian', 'traffic_cone'
        ]

        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev,
                                  ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat)
        
        loss_inputs = [
            gt_bboxes_3d[0], gt_labels_3d[0], map_gt_bboxes_3d, map_gt_labels_3d,
            outs, ego_fut_trajs, ego_fut_masks[0], ego_fut_cmd, gt_attr_labels[0]
        ]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        bbox_results = []
        for i, (bboxes, scores, labels, trajs, map_bboxes, \
                map_scores, map_labels, map_pts) in enumerate(bbox_list):
            bbox_result = bbox3d2result(bboxes, scores, labels)
            bbox_result['trajs_3d'] = trajs.cpu()
            map_bbox_result = self.map_pred2result(map_bboxes, map_scores, map_labels, map_pts)
            bbox_result.update(map_bbox_result)
            bbox_result['ego_fut_preds'] = outs['ego_fut_preds'][i].cpu()
            bbox_result['ego_fut_cmd'] = ego_fut_cmd.cpu()
            bbox_results.append(bbox_result)

        assert len(bbox_results) == 1, 'only support batch_size=1 now'
        score_threshold = 0.6
        with torch.no_grad():
            # c_bbox_results = copy.deepcopy(bbox_results)

            bbox_result = bbox_results[0]
            gt_bbox = gt_bboxes_3d[0][0]
            gt_label = gt_labels_3d[0][0].to('cpu')
            gt_attr_label = gt_attr_labels[0][0].to('cpu')
            fut_valid_flag = bool(fut_valid_flag[0][0])
            # filter pred bbox by score_threshold
            mask = bbox_result['scores_3d'] > score_threshold
            bbox_result['boxes_3d'] = bbox_result['boxes_3d'][mask]
            bbox_result['scores_3d'] = bbox_result['scores_3d'][mask]
            bbox_result['labels_3d'] = bbox_result['labels_3d'][mask]
            bbox_result['trajs_3d'] = bbox_result['trajs_3d'][mask]

            matched_bbox_result = self.assign_pred_to_gt_vip3d(
                bbox_result, gt_bbox, gt_label)

            metric_dict = self.compute_motion_metric_vip3d(
                gt_bbox, gt_label, gt_attr_label, bbox_result,
                matched_bbox_result, mapped_class_names)

            # ego planning metric
            assert ego_fut_trajs.shape[0] == 1, 'only support batch_size=1 for testing'
            ego_fut_preds = bbox_result['ego_fut_preds']
            ego_fut_trajs = ego_fut_trajs[0, 0]
            ego_fut_cmd = ego_fut_cmd[0, 0, 0]
            ego_fut_cmd_idx = torch.nonzero(ego_fut_cmd)[0, 0]
            ego_fut_pred = ego_fut_preds[ego_fut_cmd_idx]
            ego_fut_pred = ego_fut_pred.cumsum(dim=-2)
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)

            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs = ego_fut_pred[None],
                gt_ego_fut_trajs = ego_fut_trajs[None],
                gt_agent_boxes = gt_bbox,
                gt_agent_feats = gt_attr_label.unsqueeze(0),
                fut_valid_flag = fut_valid_flag
            )
            metric_dict.update(metric_dict_planner_stp3)

        return outs['bev_embed'], bbox_results, metric_dict, losses

    def simple_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        prev_bev=None,
        points=None,
        fut_valid_flag=None,
        rescale=False,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs
    ):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts, metric_dict = self.simple_test_pts(
            img_feats,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            prev_bev,
            fut_valid_flag=fut_valid_flag,
            rescale=rescale,
            start=None,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['metric_results'] = metric_dict

        return new_prev_bev, bbox_list

    def simple_test_pts(
        self,
        x,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        prev_bev=None,
        fut_valid_flag=None,
        rescale=False,
        start=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
    ):
        """Test function"""
        mapped_class_names = [
            'car', 'truck', 'construction_vehicle', 'bus',
            'trailer', 'barrier', 'motorcycle', 'bicycle', 
            'pedestrian', 'traffic_cone'
        ]

        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev,
                                  ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        bbox_results = []
        for i, (bboxes, scores, labels, trajs, map_bboxes, \
                map_scores, map_labels, map_pts) in enumerate(bbox_list):
            bbox_result = bbox3d2result(bboxes, scores, labels)
            bbox_result['trajs_3d'] = trajs.cpu()
            map_bbox_result = self.map_pred2result(map_bboxes, map_scores, map_labels, map_pts)
            bbox_result.update(map_bbox_result)
            bbox_result['ego_fut_preds'] = outs['ego_fut_preds'][i].cpu()
            bbox_result['ego_fut_cmd'] = ego_fut_cmd.cpu()
            bbox_results.append(bbox_result)

        assert len(bbox_results) == 1, 'only support batch_size=1 now'
        score_threshold = 0.6
        with torch.no_grad():
            c_bbox_results = copy.deepcopy(bbox_results)

            bbox_result = c_bbox_results[0]
            gt_bbox = gt_bboxes_3d[0][0]
            gt_label = gt_labels_3d[0][0].to('cpu')
            gt_attr_label = gt_attr_labels[0][0].to('cpu')
            fut_valid_flag = bool(fut_valid_flag[0][0])
            # filter pred bbox by score_threshold
            mask = bbox_result['scores_3d'] > score_threshold
            bbox_result['boxes_3d'] = bbox_result['boxes_3d'][mask]
            bbox_result['scores_3d'] = bbox_result['scores_3d'][mask]
            bbox_result['labels_3d'] = bbox_result['labels_3d'][mask]
            bbox_result['trajs_3d'] = bbox_result['trajs_3d'][mask]

            matched_bbox_result = self.assign_pred_to_gt_vip3d(
                bbox_result, gt_bbox, gt_label)

            metric_dict = self.compute_motion_metric_vip3d(
                gt_bbox, gt_label, gt_attr_label, bbox_result,
                matched_bbox_result, mapped_class_names)

            # ego planning metric
            assert ego_fut_trajs.shape[0] == 1, 'only support batch_size=1 for testing'
            ego_fut_preds = bbox_result['ego_fut_preds']
            ego_fut_trajs = ego_fut_trajs[0, 0]
            ego_fut_cmd = ego_fut_cmd[0, 0, 0]
            ego_fut_cmd_idx = torch.nonzero(ego_fut_cmd)[0, 0]
            ego_fut_pred = ego_fut_preds[ego_fut_cmd_idx]
            ego_fut_pred = ego_fut_pred.cumsum(dim=-2)
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)

            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs = ego_fut_pred[None],
                gt_ego_fut_trajs = ego_fut_trajs[None],
                gt_agent_boxes = gt_bbox,
                gt_agent_feats = gt_attr_label.unsqueeze(0),
                fut_valid_flag = fut_valid_flag
            )
            metric_dict.update(metric_dict_planner_stp3)

        return outs['bev_embed'], bbox_results, metric_dict

    def map_pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            map_boxes_3d=bboxes.to('cpu'),
            map_scores_3d=scores.cpu(),
            map_labels_3d=labels.cpu(),
            map_pts_3d=pts.to('cpu'))

        if attrs is not None:
            result_dict['map_attrs_3d'] = attrs.cpu()

        return result_dict

    def assign_pred_to_gt_vip3d(
        self,
        bbox_result,
        gt_bbox,
        gt_label,
        match_dis_thresh=2.0
    ):
        """Assign pred boxs to gt boxs according to object center preds in lcf.
        Args:
            bbox_result (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
        """     
        dynamic_list = [0,1,3,4,6,7,8]
        matched_bbox_result = torch.ones(
            (len(gt_bbox)), dtype=torch.long) * -1  # -1: not assigned
        gt_centers = gt_bbox.center[:, :2]
        pred_centers = bbox_result['boxes_3d'].center[:, :2]
        dist = torch.linalg.norm(pred_centers[:, None, :] - gt_centers[None, :, :], dim=-1)
        pred_not_dyn = [label not in dynamic_list for label in bbox_result['labels_3d']]
        gt_not_dyn = [label not in dynamic_list for label in gt_label]
        dist[pred_not_dyn] = 1e6
        dist[:, gt_not_dyn] = 1e6
        dist[dist > match_dis_thresh] = 1e6

        r_list, c_list = linear_sum_assignment(dist)

        for i in range(len(r_list)):
            if dist[r_list[i], c_list[i]] <= match_dis_thresh:
                matched_bbox_result[c_list[i]] = r_list[i]

        return matched_bbox_result

    def compute_motion_metric_vip3d(
        self,
        gt_bbox,
        gt_label,
        gt_attr_label,
        pred_bbox,
        matched_bbox_result,
        mapped_class_names,
        match_dis_thresh=2.0,
    ):
        """Compute EPA metric for one sample.
        Args:
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            pred_bbox (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            EPA_dict (dict): EPA metric dict of each cared class.
        """
        motion_cls_names = ['car', 'pedestrian']
        motion_metric_names = ['gt', 'cnt_ade', 'cnt_fde', 'hit',
                               'fp', 'ADE', 'FDE', 'MR']
        
        metric_dict = {}
        for met in motion_metric_names:
            for cls in motion_cls_names:
                metric_dict[met+'_'+cls] = 0.0

        veh_list = [0,1,3,4]
        ignore_list = ['construction_vehicle', 'barrier',
                       'traffic_cone', 'motorcycle', 'bicycle']

        for i in range(pred_bbox['labels_3d'].shape[0]):
            pred_bbox['labels_3d'][i] = 0 if pred_bbox['labels_3d'][i] in veh_list else pred_bbox['labels_3d'][i]
            box_name = mapped_class_names[pred_bbox['labels_3d'][i]]
            if box_name in ignore_list:
                continue
            if i not in matched_bbox_result:
                metric_dict['fp_'+box_name] += 1

        for i in range(gt_label.shape[0]):
            gt_label[i] = 0 if gt_label[i] in veh_list else gt_label[i]
            box_name = mapped_class_names[gt_label[i]]
            if box_name in ignore_list:
                continue
            gt_fut_masks = gt_attr_label[i][self.fut_ts*2:self.fut_ts*3]
            num_valid_ts = sum(gt_fut_masks==1)
            if num_valid_ts == self.fut_ts:
                metric_dict['gt_'+box_name] += 1
            if matched_bbox_result[i] >= 0 and num_valid_ts > 0:
                metric_dict['cnt_ade_'+box_name] += 1
                m_pred_idx = matched_bbox_result[i]
                gt_fut_trajs = gt_attr_label[i][:self.fut_ts*2].reshape(-1, 2)
                gt_fut_trajs = gt_fut_trajs[:num_valid_ts]
                pred_fut_trajs = pred_bbox['trajs_3d'][m_pred_idx].reshape(self.fut_mode, self.fut_ts, 2)
                pred_fut_trajs = pred_fut_trajs[:, :num_valid_ts, :]
                gt_fut_trajs = gt_fut_trajs.cumsum(dim=-2)
                pred_fut_trajs = pred_fut_trajs.cumsum(dim=-2)
                gt_fut_trajs = gt_fut_trajs + gt_bbox[i].center[0, :2]
                pred_fut_trajs = pred_fut_trajs + pred_bbox['boxes_3d'][int(m_pred_idx)].center[0, :2]

                dist = torch.linalg.norm(gt_fut_trajs[None, :, :] - pred_fut_trajs, dim=-1)
                ade = dist.sum(-1) / num_valid_ts
                ade = ade.min()

                metric_dict['ADE_'+box_name] += ade
                if num_valid_ts == self.fut_ts:
                    fde = dist[:, -1].min()
                    metric_dict['cnt_fde_'+box_name] += 1
                    metric_dict['FDE_'+box_name] += fde
                    if fde <= match_dis_thresh:
                        metric_dict['hit_'+box_name] += 1
                    else:
                        metric_dict['MR_'+box_name] += 1

        return metric_dict

    ### same planning metric as stp3
    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        fut_valid_flag
    ):
        """Compute planner metric for one sample same as stp3."""
        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
        }
        metric_dict['fut_valid_flag'] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats)
        occupancy = torch.logical_or(segmentation, pedestrian)

        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i+1)*2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy)
                metric_dict['plan_L2_{}s'.format(i+1)] = traj_L2
                metric_dict['plan_obj_col_{}s'.format(i+1)] = obj_coll.mean().item()
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = obj_box_coll.mean().item()
            else:
                metric_dict['plan_L2_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = 0.0
            
        return metric_dict

    def set_epoch(self, epoch): 
        self.pts_bbox_head.epoch = epoch

