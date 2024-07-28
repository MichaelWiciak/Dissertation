from lora import LoRA 
from lora import LoRAConfig
import torch

lora_config = LoRAConfig(
    alpha=0.1,
    beta=0.1,
    gamma=0.1,
    sigma=0.1,   
    lambda_reg=0.1,
    adaptation_steps=10, 
    device='cuda' if torch.cuda.is_available() else 'cpu'  # Device for computation
)

