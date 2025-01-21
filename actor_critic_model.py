import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticModel(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_dim=256):
        super(ActorCriticModel, self).__init__()

        self.hidden_dim = hidden_dim

        # Convolutional layers with padding to preserve dimensions
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)  # Padding added
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2)  # Padding added

        # LSTM layer
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, batch_first=True)

        # Fully connected layers for state-value and advantage-value streams
        self.fc_output_layer = nn.Linear(256, 128)
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

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
        state = F.relu(self.fc_output_layer(x))
        output = self.policy_head(state)
        value = self.value_head(state)

        return output, value, hidden_state
