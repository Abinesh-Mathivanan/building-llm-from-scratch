from tiktoken import get_encoding 
import torch.nn as nn 

tokenizer = get_encoding("gpt2")
file_path = "./data/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

# tokenized_text_data = tokenizer.encode(text_data)
# print(tokenized_text_data)

train_ratio = 0.90 
split_index = int(train_ratio * len(text_data))
# print(split_index)
train_sample = text_data[:split_index]
validate_sample = text_data[split_index:]

# train_data_loader = use PyTorchDataLoader(include params) from Chapter 2 
# validate_data_loader = use the same 


                # ---------------------------- Loss for each batch --------------------------- #

# this function calculates the loss of any given batch (individual)
def calculate_loss_batch(input_batch, output_batch, model, device):
    # input_batch = input_batch.to(device) if you use any GPU
    # output_batch = output_batch.to(device) if you use any GPU
    logits = model(input_batch)
    loss = nn.CrossEntropyLoss(
        logits.flatten(0, 1), output_batch.flatten()
    )
    return loss 


# this function returns the total loss of the data batch 
def calculate_total_loss(data_loader, model, device, num_batches = None):
    total_loss = 0 
    if not data_loader:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(len(data_loader), num_batches)
        for index, (input_batch, output_batch) in data_loader:
            if index < num_batches:
                loss = calculate_loss_batch(input_batch, output_batch, model, device)
                total_loss += loss.item()
            else:
                break 
    return total_loss / num_batches





