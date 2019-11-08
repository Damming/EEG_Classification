import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

from model import Inception_model
from data_provider import ERP_dataset


training_data = ERP_dataset('train')
validate_data = ERP_dataset('validate')

model = Inception_model()
criterion = torch.nn.MSELoss()
parameters = model.parameters()
optimizer = optim.Adam(parameters, lr=0.01, weight_decay=0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10)

loss_list = []

for epoch in range(200):

    # start training
    model.train()
    training_loader = DataLoader(training_data, batch_size=20, shuffle=True)

    train_loss = 0

    for data, label in training_loader:
        output = model(data)
        loss = criterion(output, label.float().unsqueeze(1))
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(training_loader)
    print("train epoch: {0}, loss: {1:.4f}.".format(epoch, train_loss))

    # start validate
    model.eval()
    validate_loader = DataLoader(validate_data, batch_size=1, shuffle=False)

    validate_loss = 0

    for data, label in validate_loader:
        with torch.no_grad():
            output = model(data)
        loss = criterion(output, label.float().unsqueeze(1))
        validate_loss += loss

    validate_loss = validate_loss / len(validate_loader)
    loss_list.append(validate_loss)
    print("validation epoch: {0}, loss: {1:.4f}.".format(epoch, validate_loss))

    scheduler.step(validate_loss)

# plot losses
plt.plot(range(0, len(loss_list)), loss_list, '.-')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('./training_result.png')
# save model
torch.save(model.state_dict(), "./saved_model.pth")
