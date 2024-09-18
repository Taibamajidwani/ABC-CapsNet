import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, num_routes):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.capsules = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=9, stride=2) for _ in range(num_capsules)])

    def forward(self, x):
        u_hat = [capsule(x) for capsule in self.capsules]
        u_hat_stack = torch.stack(u_hat, dim=1)
        return self.dynamic_routing(u_hat_stack)

    def dynamic_routing(self, u_hat_stack, num_iterations=3):
        batch_size = u_hat_stack.size(0)
        b_ij = torch.zeros(batch_size, self.num_capsules, u_hat_stack.size(2), 1).to(u_hat_stack.device)

        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            s_j = (c_ij * u_hat_stack).sum(dim=2, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                b_ij = b_ij + (u_hat_stack * v_j).sum(dim=-1, keepdim=True)

        return v_j.squeeze(-1)

    def squash(self, s_j):
        s_j_norm = torch.norm(s_j, dim=-1, keepdim=True)
        return (s_j_norm / (1.0 + s_j_norm ** 2)) * s_j

class CapsuleNetwork1(nn.Module):
    def __init__(self):
        super(CapsuleNetwork1, self).__init__()
        self.primary_capsules = CapsuleLayer(num_capsules=8, in_channels=256, out_channels=8, num_routes=3)

    def forward(self, x):
        return self.primary_capsules(x)

class CapsuleNetwork2(nn.Module):
    def __init__(self):
        super(CapsuleNetwork2, self).__init__()
        self.secondary_capsules = CapsuleLayer(num_capsules=10, in_channels=8, out_channels=16, num_routes=3)
        self.decoder = nn.Linear(16 * 10, 1)

    def forward(self, x):
        x = self.secondary_capsules(x)
        return self.decoder(x.view(x.size(0), -1))
