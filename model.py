import torch
import torch.nn as nn
import torch.nn.functional as F


class DDQNLSTM(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DDQNLSTM, self).__init__()

        # Convolutional layers with padding to preserve dimensions
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)  # Padding added
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2, padding=1)  # Padding added

        # LSTM layer
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, batch_first=True)

        # Fully connected layers for state-value and advantage-value streams
        self.fc_state = nn.Linear(256, 128)
        self.state_value = nn.Linear(128, 1)

        self.fc_advantage = nn.Linear(256, 128)
        self.advantage_values = nn.Linear(128, n_actions)

    def forward(self, x, hidden_state=None):
        batch_size = x.size(0)

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten for LSTM input
        x = x.view(batch_size, -1)  # Flatten all but the batch dimension
        x = x.unsqueeze(1)  # Add time dimension for LSTM (sequence length = 1)

        # LSTM layer
        if hidden_state is None:
            x, hidden_state = self.lstm(x)
        else:
            x, hidden_state = self.lstm(x, hidden_state)

        x = x.squeeze(1)  # Remove the time dimension

        # State-value stream
        state = F.relu(self.fc_state(x))
        state_value = self.state_value(state)

        # Advantage-value stream
        adv = F.relu(self.fc_advantage(x))
        advantage_values = self.advantage_values(adv)

        # Combine state value and advantage values into Q-values
        q_values = state_value + (advantage_values - advantage_values.mean(dim=1, keepdim=True))

        return q_values, hidden_state
