import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class SimpleSSM(nn.Module):
    def __init__(self, state_dim=1):
        super().__init__()
        self.A = nn.Parameter(torch.tensor([0.9]))  # State decay rate
        self.B = nn.Parameter(torch.tensor([1.0]))  # Input weight
        self.C = nn.Parameter(torch.tensor([1.0]))  # State-to-output mapping

    def forward(self, x):
        batch, seq_len = x.shape
        h = torch.zeros(batch, self.A.shape[0])  # Initial state
        outputs = []
        states = []

        for t in range(seq_len):
            h = self.A * h + self.B * x[:, t:t+1]  # State update
            y = self.C * h  # Output
            outputs.append(y)
            states.append(h)
        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, state_dim)
        states = torch.cat(states, dim=1)    # (batch, seq_len, state_dim)
        return outputs, states
    
def generate_data(seq_len=100):
    t = torch.linspace(0, 4 * torch.pi, seq_len)
    x = torch.sin(t) + 0.1 * torch.randn(seq_len)  # Noisy sine wave
    return x.unsqueeze(0)  # (1, seq_len)


model = SimpleSSM(state_dim=1)
x = generate_data(seq_len=50)

with torch.no_grad():
    y_pred, states = model(x)

print("Predicted Output y(t):", y_pred.squeeze().numpy())
print("Hidden States h(t):", states.squeeze().numpy())

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