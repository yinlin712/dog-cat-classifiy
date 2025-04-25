import torch

def save_model(model, filename='model.pth'):
    torch.save(model.state_dict(), filename)

def load_model(model, filename='model.pth'):
    model.load_state_dict(torch.load(filename))
    return model
