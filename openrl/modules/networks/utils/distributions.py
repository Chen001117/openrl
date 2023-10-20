import torch
import torch.nn as nn
import math

from .util import init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#


# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        # return super().log_prob(actions).sum(-1, keepdim=True)
        return super().log_prob(actions)

    def entropy(self):
        return super().entropy()

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, action_masks=None):
        x = self.linear(x)
        if action_masks is not None:
            x[action_masks == 0] = -1e10
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(DiagGaussian, self).__init__()

        self._hidden_size = 64
        
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = nn.Sequential(
            nn.Linear(num_inputs, self._hidden_size), 
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, num_outputs),
            nn.Tanh()
        )
        self.logstd = nn.Sequential(
            nn.Linear(num_inputs, self._hidden_size), 
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, num_outputs),
            nn.Tanh()
        )
        
        self._action_low = torch.tensor([[-0.1, -0.05, -math.pi/12]])
        self._action_high = torch.tensor([[0.5, 0.05, math.pi/12]])
        self._logstd_low = torch.log(torch.tensor([[0.06, 0.01, math.pi/60.]]))
        self._logstd_high = torch.log(torch.tensor([[0.6, 0.1, math.pi/6.]]))
        self._first_time = True
        # self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        # self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        
        if self._first_time:
            self._first_time = False
            self._action_low = self._action_low.to(x.device)
            self._action_high = self._action_high.to(x.device)
            self._logstd_low = self._logstd_low.to(x.device)
            self._logstd_high = self._logstd_high.to(x.device)
        
        action_mean = self.fc_mean(x)
        action_mean = (action_mean+1) / 2
        action_mean *= (self._action_high-self._action_low)
        action_mean += self._action_low
        
        action_logstd = self.logstd(x)
        action_logstd = (action_logstd+1) / 2
        action_logstd *= (self._logstd_high-self._logstd_low)
        action_logstd += self._logstd_low
        
        return FixedNormal(action_mean, action_logstd.exp())
        

        # #  An ugly hack for my KFAC implementation.
        # # zeros = torch.zeros(action_mean.size())
        # zeros = torch.zeros_like(action_mean)

        # # if x.is_cuda:
        # #     zeros = zeros.cuda()

        # action_logstd = self.logstd(zeros)
        # return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
