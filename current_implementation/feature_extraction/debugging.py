import torch

def quicksave(a):
    torch.save(a, '/tmp/debugging_file.pt')

def quickcompare(a):
    b = torch.load('/tmp/debugging_file.pt', map_location=a.device)
    if a.shape != b.shape: 
        print('Shape mismatch:')
        print(a.shape)
        print(b.shape)
    else:
        print(f'Summed diff: {torch.abs(a-b).sum()}')
        print(f'Mean diff: {torch.abs(a-b).mean()}')