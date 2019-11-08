import torch
from torch.utils.data.dataloader import DataLoader

from model import Inception_model
from data_provider import ERP_dataset


test_data = ERP_dataset('train')

model = Inception_model()
saved_states = torch.load("./saved_model.pth")
model.load_state_dict(saved_states)
criterion = torch.nn.MSELoss()

model.eval()
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

accuracies = 0

for data, label in test_loader:
    with torch.no_grad():
        output = model(data)

    loss = float(criterion(output, label.float().unsqueeze(1)))
    if loss > 1:
        accuracy = 0
    else:
        accuracy = 1 - loss
    accuracies += accuracy

print("test accuracy: {0:.4f}.".format(accuracies / len(test_loader)))
