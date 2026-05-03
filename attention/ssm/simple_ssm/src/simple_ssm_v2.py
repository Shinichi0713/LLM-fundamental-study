import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(42)

class SimpleSSM(nn.Module):
    def __init__(self, state_dim=1):
        super().__init__()
        self.state_dim = state_dim
        # Parameters (fixed values)
        self.A = nn.Parameter(torch.tensor([0.9]))  # State decay rate
        self.B = nn.Parameter(torch.tensor([1.0]))  # Input weight
        self.C = nn.Parameter(torch.tensor([1.0]))  # State-to-output mapping

    def forward(self, x):
        # x: (batch, seq_len)
        batch, seq_len = x.shape
        h = torch.zeros(batch, self.state_dim)  # Initial state
        outputs = []
        states = []

        for t in range(seq_len):
            # State update: h_t = A * h_{t-1} + B * x_t
            h = self.A * h + self.B * x[:, t:t+1]
            # Output: y_t = C * h_t
            y = self.C * h
            outputs.append(y)
            states.append(h)

        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, 1)
        states = torch.cat(states, dim=1)    # (batch, seq_len, 1)
        return outputs, states

# Data generation: Sine wave + Noise
def generate_data(seq_len=100):
    t = torch.linspace(0, 4 * torch.pi, seq_len)
    x = torch.sin(t) + 0.1 * torch.randn(seq_len)  # Noisy sine wave
    return x.unsqueeze(0)  # (1, seq_len)

# Model and data initialization
model = SimpleSSM(state_dim=1)
x = generate_data(seq_len=50)

# Forward pass
with torch.no_grad():
    y_pred, states = model(x)

# Visualization
plt.figure(figsize=(12, 4))

# Input vs. Output comparison
plt.subplot(1, 2, 1)
plt.plot(x.squeeze().numpy(), label='Input x(t)')
plt.plot(y_pred.squeeze().numpy(), label='Output y(t)')
plt.title('SSM Input and Output')
plt.legend()

# State transition
plt.subplot(1, 2, 2)
plt.plot(states.squeeze().numpy(), label='State h(t)', color='red')
plt.title('SSM Internal State')
plt.legend()

plt.tight_layout()
plt.show()