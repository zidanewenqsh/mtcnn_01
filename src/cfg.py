import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THREHOLD = {
    12: [0.9, 0.3],
    24: [0.99, 0.3],
    48: [0.99, 0.3]
}
