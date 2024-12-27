import torch 
from torch.utils.data import Dataset, DataLoader
from bytePair import tokenizer

class PytorchTokenizer:
    def __init__(self, text, max_length, stride):
        self.input_ids = []
        self.output_ids = []

        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(token_ids[i : i + max_length])
            self.output_ids.append(token_ids[i+1: i + max_length + 1])

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return torch.tensor(self.input_ids[index]), torch.tensor(self.output_ids[index])
    
def PytorchDataLoader(text, batch_size = 4, max_length = 256, stride = 128, shuffle = True, num_workers = 0):
    dataset = PytorchTokenizer(text, max_length, stride)
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)

with open("data/the-verdict.txt", "r", encoding="utf-8") as file:
    text = file.read()
data_loader = PytorchDataLoader(text, batch_size=1, max_length=4, shuffle=False, stride=1)
length_data = len(data_loader)
data_iter = iter(data_loader)
next_data = next(data_iter)
print("Length of data:", length_data)
print("Data Batch:", next_data)
        
    
