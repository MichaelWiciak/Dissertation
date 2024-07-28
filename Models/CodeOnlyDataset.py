
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from transformers import BertTokenizer, BertForMaskedLM
from transformers import get_linear_schedule_with_warmup
# import mw_utils.py, it is found in parent directory
import sys
sys.path.append("..")
import mw_utils as mw
import json
from torch.nn.utils.rnn import pad_sequence

class CodeOnlyCompletionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = [json.loads(line) for line in data]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Assuming each example contains 'code_tokens' and 'output'
        code_tokens = sample['code_tokens']
        output = sample['output']

        # Convert code_tokens to token IDs using tokenizer
        input_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)

        # Create mask for the masked token
        mask_index = code_tokens.index('<mask>')
        attention_mask = [1] * len(input_ids)
        attention_mask[mask_index] = 0  # Set the masked token's attention to 0

        # Convert output to token ID
        label_id = self.tokenizer.convert_tokens_to_ids(output)

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'label': torch.tensor(label_id)
        }


