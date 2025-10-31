from typing import Dict, List
from dataclasses import dataclass

@dataclass
class DDPM_Config:
    Channels: List[int]
    Attentions: List[bool]
    Upscales: List[bool]
    num_groups: int 
    dropout_prob: float 
    num_heads: int 
    input_channels: int 
    output_channels: int 
    time_steps: int 

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)