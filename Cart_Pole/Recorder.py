from collections import deque
from typing import Optional, Iterable
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Recorder(deque):
    def __init__(self, iterable: Optional[Iterable]=None, maxlen: Optional[int]=None):
        if iterable is None:
            super(Recorder, self).__init__(maxlen=maxlen)
        else:
            super(Recorder, self).__init__(iterable, maxlen=maxlen)

    def tensor(self):
        """ Give [B, C, H, W] tensor for images 
        e.g. [H, W]x4 -> [4, H, W]
        """
        return torch.stack(tuple(torch.tensor(x, device=device) for x in self)).to(device)
    
    def numpy(self)->np.ndarray:
        return self.tensor().cpu().numpy()