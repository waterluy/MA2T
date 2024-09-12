import math
import torch

class ADV:
    def __init__(self, 
                 img: bool=False,
                 agent: bool=False,
                 map: bool=False,
                 mode: str='miloo',
                 ) -> None:
        self.attack_img = img
        self.attack_agent = agent
        self.attack_map = map

        self.noise_img = None
        self.noise_agent = None
        self.noise_map = None

        self.random_start = False

        self.mode = mode

        self.steps = 0
        self.eps_img = 8 # / 255
        self.alpha_img = 1 # / 255
        self.eps_agent = 0.06
        self.alpha_agent = 0.009
        self.eps_map = 0.025
        self.alpha_map = 0.004

        self.mi_noise_img_grad = 0
        self.mi_noise_agent_grad = 0
        self.mi_noise_map_grad = 0

        self.miu = 0.2
        
        # self.set_alpha()
        
        
    def set_alpha(self):
        # self.alpha_img = self.eps_img / math.sqrt(self.steps)
        # self.alpha_agent = self.eps_agent / math.sqrt(self.steps)
        # self.alpha_map = self.eps_map / math.sqrt(self.steps)
        if self.steps > 0:
            self.alpha_img = self.eps_img / self.steps
            self.alpha_agent = self.eps_agent / self.steps
            self.alpha_map = self.eps_map / self.steps
    

    def update_cfg(self, 
                   new_cfg: dict):
        for key, value in new_cfg.items():
            if getattr(self, key) is not None:
                setattr(self, key, value)
                print( key, value)
            else:
                print(f"wrong cfg key: {key}")
        self.set_alpha()
    

    def update_noise(self, loss, ):
        ob = ['img', 'map', 'agent']
        noises_require_grad = []
        for o in ob:
            noise = getattr(self, f'noise_{o}') 
            attack  = getattr(self, f'attack_{o}')
            if attack:
                assert noise.requires_grad
                noises_require_grad.append(noise)

        if 0 != len(noises_require_grad):
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=noises_require_grad,
                # allow_unused=False,
            )

            index = 0
            for o in ob:
                noise = getattr(self, f'noise_{o}') 
                attack  = getattr(self, f'attack_{o}')
                if attack:
                    noise_grad = grads[index]
                    assert noise_grad is not None
                    # assert not noise_grad.detach().cpu().equal(torch.zeros(noise_grad.shape))
                    
                    alpha = getattr(self, f'alpha_{o}')
                    if self.mode == 'loo':
                        noise = noise + alpha * noise_grad.sign()
                    elif self.mode == 'l1':
                        noise = noise + alpha * noise_grad / torch.norm(noise_grad, p=1, dim=None, keepdim=False, out=None, dtype=None)
                    elif self.mode == 'l2':
                        noise = noise + alpha * noise_grad / torch.norm(noise_grad, p=2, dim=None, keepdim=False, out=None, dtype=None)
                    elif self.mode == 'miloo':
                        mi_noise_grad = getattr(self, f'mi_noise_{o}_grad')
                        mi_noise_grad = self.miu * mi_noise_grad + (1 - self.miu) * noise_grad
                        noise = noise + alpha * mi_noise_grad.sign()
                        setattr(self, f'mi_noise_{o}_grad', mi_noise_grad)
                    else:
                        raise Exception('mode: ', self.mode)
                    
                    noise = noise.detach()
                    noise.requires_grad_(True)
                    setattr(self, f'noise_{o}', noise)
                    
                    index = index + 1 

    def update_noise_task(self, detection_loss, map_loss, motion_loss, plan_loss):
        ob = ['img', 'map', 'agent']
        noises_require_grad = []
        for o in ob:
            noise = getattr(self, f'noise_{o}') 
            attack  = getattr(self, f'attack_{o}')
            if attack:
                assert noise.requires_grad
                noises_require_grad.append(noise)
        assert len(noises_require_grad) == len(ob)

        img_loss = detection_loss + map_loss + motion_loss
        map_loss = plan_loss
        agent_loss = plan_loss
        
        noise_grads = torch.autograd.grad(
            outputs=[img_loss, map_loss, agent_loss],
            inputs=noises_require_grad,
            # allow_unused=False,
            retain_graph=False,
        )
        
        for i, o in enumerate(ob):
            alpha = getattr(self, f'alpha_{o}')
            noise = getattr(self, f'noise_{o}') 
            noise_grad = noise_grads[i]
            assert noise_grad is not None
            if self.mode == 'loo':
                noise = noise + alpha * noise_grad.sign()
            elif self.mode == 'l1':
                noise = noise + alpha * noise_grad / torch.norm(noise_grad, p=1, dim=None, keepdim=False, out=None, dtype=None)
            elif self.mode == 'l2':
                noise = noise + alpha * noise_grad / torch.norm(noise_grad, p=2, dim=None, keepdim=False, out=None, dtype=None)
            elif self.mode == 'miloo':
                mi_noise_grad = getattr(self, f'mi_noise_{o}_grad')
                mi_noise_grad = self.miu * mi_noise_grad + (1 - self.miu) * noise_grad
                noise = noise + alpha * mi_noise_grad.sign()
                setattr(self, f'mi_noise_{o}_grad', mi_noise_grad)
            else:
                raise Exception('mode: ', self.mode)
        
            noise = noise.detach()
            noise.requires_grad_(True)
            setattr(self, f'noise_{o}', noise)

