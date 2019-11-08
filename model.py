import torch
import torch.nn as nn

class Inception_model(nn.Module):

    def __init__(self):
        super(Inception_model, self).__init__()

        self.time_domain_conv = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(42, 1), padding=(10, 0))
        self.norm1 = nn.BatchNorm2d(num_features=6)
        self.space_domain_conv = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(1, 20))
        self.norm2 = nn.BatchNorm2d(num_features=12)
        self.activation1 = nn.ReLU(inplace=True)
        self.pooling = nn.AvgPool2d(kernel_size=(4, 1), stride=4)

        # inception block branch 1
        self.inception_branch_1_conv_1 = nn.Conv2d(in_channels=12, out_channels=4, kernel_size=1)
        self.inception_branch_1_norm1 = nn.BatchNorm2d(num_features=4)
        self.inception_branch_1_conv_2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(5, 1), padding=(2, 0))
        self.inception_branch_1_norm2 = nn.BatchNorm2d(num_features=4)

        # inception block branch 2
        self.inception_branch_2_conv_1 = nn.Conv2d(in_channels=12, out_channels=4, kernel_size=1)
        self.inception_branch_2_norm1 = nn.BatchNorm2d(num_features=4)
        self.inception_branch_2_conv_2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(7, 1), padding=(3, 0))
        self.inception_branch_2_norm2 = nn.BatchNorm2d(num_features=4)

        # inception block branch 3
        self.inception_branch_3_pooling = nn.MaxPool2d(kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.inception_branch_3_conv = nn.Conv2d(in_channels=12, out_channels=4, kernel_size=(1, 1))
        self.inception_branch_3_norm = nn.BatchNorm2d(num_features=4)

        self.down_sample = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(4, 1), stride=4)
        self.norm3 = nn.BatchNorm2d(num_features=12)
        self.dropout = nn.Dropout2d(0.5, inplace=True)
        self.fully_connection = nn.Linear(in_features=60, out_features=1)
        self.activation2 = nn.Sigmoid()

        # initialize parameters
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.uniform_(layer.weight)

    def forward(self, input):

        # input: size: [20, 1, 101, 20], type: float32 (float 64 raises error)
        pre_inception = self.time_domain_conv(input)  # size: [20, 6, 80, 20]
        pre_inception = self.norm1(pre_inception)
        pre_inception = self.space_domain_conv(pre_inception)  # size: [20, 12, 80, 1]
        pre_inception = self.norm2(pre_inception)
        pre_inception = self.activation1(pre_inception)  # size: [20, 12, 20, 1]
        pre_inception = self.pooling(pre_inception)

        inception_branch_1 = self.inception_branch_1_conv_1(pre_inception)  # size: [20, 4, 20, 1]
        inception_branch_1 = self.inception_branch_1_norm1(inception_branch_1)
        inception_branch_1 = self.inception_branch_1_conv_2(inception_branch_1)  # size: [20, 4, 20, 1]
        inception_branch_1 = self.inception_branch_1_norm2(inception_branch_1)

        inception_branch_2 = self.inception_branch_2_conv_1(pre_inception)  # size: [20, 4, 20, 1]
        inception_branch_2 = self.inception_branch_2_norm1(inception_branch_2)
        inception_branch_2 = self.inception_branch_2_conv_2(inception_branch_2)  # size: [20, 4, 20, 1]
        inception_branch_2 = self.inception_branch_2_norm2(inception_branch_2)

        inception_branch_3 = self.inception_branch_3_pooling(pre_inception)  # size: [20, 12, 20, 1]
        inception_branch_3 = self.inception_branch_3_conv(inception_branch_3)  # size: [20, 4, 20, 1]
        inception_branch_3 = self.inception_branch_3_norm(inception_branch_3)

        post_inception = torch.cat((inception_branch_1, inception_branch_2, inception_branch_3), 1)  # size: [20, 12, 20, 1]
        post_inception = self.down_sample(post_inception)  # size: [20, 12, 5, 1]
        post_inception = self.norm3(post_inception)
        post_inception = self.dropout(post_inception)  # size: [20, 12, 5, 1]
        post_inception = post_inception.view(-1, 60)  # size: [20, 60]
        post_inception = self.fully_connection(post_inception)  # size: [20, 1]
        post_inception = self.activation2(post_inception)  # size: [20, 1]

        return post_inception
