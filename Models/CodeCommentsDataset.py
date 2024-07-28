
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

from transformers import AutoTokenizer, AutoModel
from loraSetup import LoRA
from loraSetup import LoRAConfig
import torch.nn as nn


# now that I have the data, I can start to build the model
# need to try prepare and train the model



lora_config = LoRAConfig(
    alpha=0.1,
    beta=0.1,
    gamma=0.1,
    sigma=0.1,   
    lambda_reg=0.1,
    adaptation_steps=10,  # Number of adaptation steps
    device='cuda' if torch.cuda.is_available() else 'cpu'  # Device for computation
)

class DataLoaders:
    def __init__(self, train_dataloader, valid_dataloader, test_dataloader):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

    # setters
    def setTrainDataloader(self, train_dataloader):
        self.train_dataloader = train_dataloader
    
    def setValidDataloader(self, valid_dataloader):
        self.valid_dataloader = valid_dataloader
    
    def setTestDataloader(self, test_dataloader):
        self.test_dataloader = test_dataloader
    
    # getters
    def getTrainDataloader(self):
        return self.train_dataloader

    def getValidDataloader(self):
        return self.valid_dataloader
    
    def getTestDataloader(self):
        return self.test_dataloader
        
    def getDataloaders(self):
        return self.train_dataloader, self.valid_dataloader, self.test_dataloader

class CodeDocstringCompletionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = [json.loads(line) for line in data]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Assuming each example contains 'code_tokens', 'docstring_tokens', and 'output'
        code_tokens = sample['code_tokens']
        docstring_tokens = sample['docstring_tokens']
        output = sample['output']

        # Convert code_tokens and docstring_tokens to token IDs using tokenizer
        code_input_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)
        docstring_input_ids = self.tokenizer.convert_tokens_to_ids(docstring_tokens)

        # Concatenate code_input_ids and docstring_input_ids
        input_ids = code_input_ids + docstring_input_ids

        # Create mask for the masked token (assuming it's in the code_tokens part)
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


    


tokeniser = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")

lora =  LoRA(model, tokeniser, config=lora_config)

dataLoader = DataLoaders(None, None, None)


class BimodalModel(nn.Module):
    def __init__(self, model_name):
        super(BimodalModel, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 50265)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        logits = self.classifier(last_hidden_state[:, 0, :])  # Use CLS token for classification
        return logits


def main():
    # load the training data, validation data and test data
    # it is found in ../DatasetManipulations/preprocesssed/[test, train, valid]/[test, train, valid]_code.jsonl
    train_data, valid_data, test_data = loadData()

    # create the data loader
    train_dataloader, valid_dataloader, test_dataloader = createDataLoader(train_data, valid_data, test_data)

    # set the data loader
    dataLoader.setTrainDataloader(train_dataloader)
    dataLoader.setValidDataloader(valid_dataloader)
    dataLoader.setTestDataloader(test_dataloader)

    # train the model
    train()

    # evaluate the model
    evaluation()





   
def train(patience=10):
    # load the pre-trained BERT model
    model_name = 'microsoft/codeBERT-base'
    # model = BertForMaskedLM.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model = model.to(lora_config.device)
    model = LoRA(model, tokeniser, config=lora_config)

    # Define your loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # set dataloaders
    train_dataloader = dataLoader.getTrainDataloader()

    num_epochs = 30
    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * num_epochs, gamma=0.1)

    counter_epoch = 1
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        counter_batch = 1
        for batch in train_dataloader:
            print(f'Epoch {counter_epoch}/{num_epochs}, Batch {counter_batch}/{len(train_dataloader)}')
            inputs = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

            # reshape the labels so that it is the same shape as the inputs
            labels = labels.view(-1)

            optimizer.zero_grad()

            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Optional: gradient clipping
            optimizer.step()
            scheduler.step()

            counter_batch += 1
            print("Loss: ", loss.item())

        counter_epoch += 1

        lora.apply(model)

        average_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}')

        # Check for early stopping
        if average_loss < best_loss:
            best_loss = average_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Validation loss has not improved for {patience} epochs. Early stopping...')
                break
    
    # Save the model
    model.save_pretrained('BERT_CODE_ONLY_COMPLETION_MODEL')
    tokeniser.save_pretrained('BERT_CODE_ONLY_COMPLETION_MODEL')


def evaluation():
    # load the model, 
    model_name = "CodeComments"
    model_directory = model_name
    model = BertForMaskedLM.from_pretrained(model_directory)
    tokeniser = BertTokenizer.from_pretrained(model_directory)



    valid_dataloader = dataLoader.getValidDataloader()

    # Put the model in evaluation mode
    model.eval()

    # Initialize variables for evaluation
    total_loss = 0.0
    num_correct = 0
    num_samples = 0

    #  Evaluate on the validation set

    with torch.no_grad():
        batch_counter = 1
        for batch in valid_dataloader:
            print(f'Batch {batch_counter}/{len(valid_dataloader)}')
            inputs = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            # reshape the labels so that it is the same shape as the inputs
            labels = labels.view(-1)

            # print all shapes of everything
            if batch_counter == 1:
                print("Input Shape: ", inputs.shape)
                print("Attention Mask Shape: ", attention_mask.shape)
                print("Labels Shape: ", labels.shape)
                # print out the inputs, attention mask and labels for just the first batch
                print("Inputs: ", inputs)
                print("Attention Mask: ", attention_mask)
                print("Labels: ", labels)

            
            # Forward pass
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # Accumulate loss
            total_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            num_correct += torch.sum(predictions == labels).item()
            num_samples += len(labels)

    # Calculate average loss and accuracy
    average_loss = total_loss / len(valid_dataloader)
    accuracy = num_correct / num_samples

    print(f'Validation Loss: {average_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

    # Put the model in evaluation mode
    model.eval()

    # Example input with a masked token
    input_text = "def get_metadata(self <mask> entity_type, entity_id):"

    # Tokenize the input
    input_ids = tokeniser.encode(input_text, return_tensors="pt")

    # Generate predictions
    with torch.no_grad():
        outputs = model(input_ids)

    # Get the predicted token IDs for the masked position
    predicted_token_id = torch.argmax(outputs.logits[0, -1]).item()

    # Convert the predicted token ID back to the actual token
    predicted_token = tokeniser.convert_ids_to_tokens(predicted_token_id)

    # Print the predicted token
    print(f"Predicted Token: {predicted_token}")
    # was expecting ,
    print(f"Expected Token: ','")



def collate_fn(batch):

    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['label'] for item in batch]

    max_length = 1 + max([len(ids) for ids in input_ids])

    # Pad input_ids and attention_mask
    input_ids_padded = pad_sequence([torch.nn.functional.pad(ids, (0, max_length - len(ids))) for ids in input_ids], batch_first=True, padding_value=tokeniser.pad_token_id)
    attention_mask_padded = pad_sequence([torch.nn.functional.pad(mask, (0, max_length - len(mask))) for mask in attention_mask], batch_first=True, padding_value=0)

    # Stack labels
    labels = torch.stack(labels).squeeze()

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'label': labels
    }

def loadData():
    # load the training data
    train_data = mw.openFile("../DatasetManipulations/preprocessed/train/train_codeComments.jsonl")
    # load the validation data
    valid_data = mw.openFile("../DatasetManipulations/preprocessed/valid/valid_codeComments.jsonl")
    # load the test data
    test_data = mw.openFile("../DatasetManipulations/preprocessed/test/test_codeComments.jsonl")

    return train_data, valid_data, test_data

def createDataLoader(train_data, valid_data, test_data):
    # Assuming you have separate datasets for train_data, valid_data, and test_data
    train_dataset = CodeDocstringCompletionDataset(train_data)
    valid_dataset = CodeDocstringCompletionDataset(valid_data)
    test_dataset = CodeDocstringCompletionDataset(test_data)

    # Set batch size
    batch_size = 26

    # Create DataLoader for training
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Create DataLoader for validation and testing
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)


    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == "__main__":
    main()