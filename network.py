import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    
    def __init__(self, n, m, num_embeddings, num_features, num_policies, num_res_blocks, mask):
        super().__init__()

        self.num_res_blocks = num_res_blocks
        self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32).view(1, 1, n, m))
        self.embedding = nn.Embedding(num_embeddings, num_features)
        #conv(relu(bn(conv(relu(bn(x))))))

        for i in range(1, num_res_blocks + 1):
            setattr(self, 'res{}_bn1'.format(i), nn.BatchNorm2d(num_features))
            setattr(self, 'res{}_conv1'.format(i), nn.Conv2d(num_features, num_features, 3, stride=1, padding=1))
            setattr(self, 'res{}_bn2'.format(i), nn.BatchNorm2d(num_features))
            setattr(self, 'res{}_conv2'.format(i), nn.Conv2d(num_features, num_features, 3, stride=1, padding=1))
        
        self.bn_post_res = nn.BatchNorm2d(num_features)

        self.conv_p = nn.Conv2d(num_features, 4, 1, stride=1, padding=0)
        self.bn_p = nn.BatchNorm2d(4)
        self.fc_p = nn.Linear(4 * n * m, num_policies)

        self.conv_v = nn.Conv2d(num_features, 1, 1, stride=1, padding=0)
        self.bn_conv_v = nn.BatchNorm2d(1)
        self.fc1_v = nn.Linear(1 * n * m, num_features * 4)
        self.fc2_v = nn.Linear(num_features * 4, 1)


    def forward(self, input, use_softmax=True):
        # input[N, H, W]
        x = self.embedding(input).permute(0, 3, 1, 2) * self.mask
        # x[N, C, H, W]

        for i in range(1, self.num_res_blocks + 1):
            y = x
            y = getattr(self, 'res{}_bn1'.format(i))(y) * self.mask
            y = torch.relu(y)
            y = getattr(self, 'res{}_conv1'.format(i))(y) * self.mask
            y = getattr(self, 'res{}_bn2'.format(i))(y) * self.mask
            y = torch.relu(y)
            y = getattr(self, 'res{}_conv2'.format(i))(y) * self.mask
            x += y

        x = torch.relu(self.bn_post_res(x) * self.mask)

        p = torch.relu(self.bn_p(self.conv_p(x) * self.mask) * self.mask)
        p = p.view(p.size(0), -1)
        p = self.fc_p(p)
        if use_softmax:
            p = torch.softmax(p, dim=1)

        v = torch.relu(self.bn_conv_v(self.conv_v(x) * self.mask) * self.mask)
        v = v.view(v.size(0), -1)
        v = torch.relu(self.fc1_v(v))
        v = torch.tanh(self.fc2_v(v)) * 0.5 + 0.5
        v = v.view(-1)

        return p, v

