import math

class PGDCfg:
    def __init__(self, 
                 img: bool=False,
                 bev: bool=False,
                 track_motion: bool=False,
                 seg_motion: bool=False,
                 motion_occ: bool=False,
                 motion_plan: bool=False) -> None:
        self.attack_img = img
        self.attack_track_motion = track_motion
        self.attack_seg_motion = seg_motion
        self.attack_motion_occ = motion_occ
        self.attack_motion_plan = motion_plan

        self.random_start = False

        self.steps = 10
        self.img_eps = 8 # / 255
        self.img_alpha = 1 # / 255
        self.track_eps = 0.025
        self.track_alpha = 0.004
        self.seg_eps = 0.07
        self.seg_alpha = 0.009
        self.motion_track_eps = 0.02
        self.motion_track_alpha = 0.004
        self.motion_traj_eps = 0.08
        self.motion_traj_alpha = 0.01 
        
        self.alpha = 0.001
        self.miu = 0.1
        
        self.set_alpha()
        
        
    def set_alpha(self):
        # self.img_alpha = self.img_eps / math.sqrt(self.steps)
        # self.track_alpha = self.track_eps / math.sqrt(self.steps)
        # self.seg_alpha = self.seg_eps / math.sqrt(self.steps)
        # self.motion_track_alpha = self.motion_track_eps / math.sqrt(self.steps)
        # self.motion_traj_alpha = self.motion_traj_eps / math.sqrt(self.steps)
        self.img_alpha = self.img_eps / self.steps
        self.track_alpha = self.track_eps / self.steps
        self.seg_alpha = self.seg_eps / self.steps
        self.motion_track_alpha = self.motion_track_eps / self.steps
        self.motion_traj_alpha = self.motion_traj_eps / self.steps
    
    def update_cfg(self, 
                   new_cfg: dict):
        for key, value in new_cfg.items():
            if getattr(self, key) is not None:
                setattr(self, key, value)
                print( key, value)
            else:
                print(f"wrong cfg key: {key}")
        self.set_alpha()
    