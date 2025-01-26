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


                # ------------------- calculating total loss over n epochs ------------------- #

def train_model(model, train_data_loader, validate_data_loader, optimizer, num_epochs, eval_freq, eval_iteration, start, tokenizer):
    train_losses, value_losses, tokens_visited = [], [], []
    token_mark, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_data_loader:
            optimizer.zero_grad()
            loss = calculate_loss_batch(input_batch, target_batch, model)
            loss.backward()
            optimizer.step()
            token_mark += input_batch.numel()
            global_step += 1 

            if global_step % eval_freq == 0:
                # train_loss, value_loss = evaluate_model() 
                #TODO: i forgot to implement the eval model, ig. i'd just leave it as an exercide to the reader.
                # you could visit sebastian's official repo for full implementation
                #TODO: else, implement evaluate_model(model, train_data_loader, value_data_loader, device, eval_iteration)
                train_loss, value_loss = -1, -1
                train_losses.append(train_loss)
                value_losses.append(value_loss)
                tokens_visited.append(token_mark)

    return train_losses, value_losses, tokens_visited
    
    # in this code, i skipped the intermediate printing statements 

    





