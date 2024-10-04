import torch

class MaskedSoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_sequence, actual_lengths):

        if actual_lengths.device != 'cpu':
            actual_lengths = actual_lengths.to('cpu')
        
        # print(batch_sequence.device)
        # print(actual_lengths.device)
        
        # Add -100000 after each element in news_title_actual_lengths
        mask = torch.arange(batch_sequence.size(1)).expand(batch_sequence.size(0), -1) >= actual_lengths.view(-1, 1)
        batch_sequence[mask] -= 100000000
    
        # Apply softmax
        return torch.nn.Softmax(dim=-1)(batch_sequence)