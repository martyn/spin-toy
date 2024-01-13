import torch
import torch.nn as nn
import torch.optim as optim

# A dummy dataset, replace with actual data
N = 100  # Number of samples
T = 100  # Number of iterations
input_dim = 10  # Input feature dimension
output_dim = 1  # Output dimension, for binary classification

# Define the LLM model as before
class SimpleLLM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLLM, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Define the custom loss function as per the algorithm
def spin_loss(model, opponent, x, y_true, y_synthetic):
    y_pred = model(x)

    # Compute log probabilities using current model parameters
    log_prob_true_current = torch.log(y_pred) * y_true + torch.log(1 - y_pred) * (1 - y_true)
    log_prob_synthetic_current = torch.log(y_pred) * y_synthetic + torch.log(1 - y_pred) * (1 - y_synthetic)

    # Compute log probabilities using previous model parameters (theta_t)
    with torch.no_grad():
        y_pred_t = opponent(x)
        log_prob_true_previous = torch.log(y_pred_t) * y_true + torch.log(1 - y_pred_t) * (1 - y_true)
        log_prob_synthetic_previous = torch.log(y_pred_t) * y_synthetic + torch.log(1 - y_pred_t) * (1 - y_synthetic)

    # Compute the loss as per the algorithm
    loss = (log_prob_true_current - log_prob_true_previous) - (log_prob_synthetic_current - log_prob_synthetic_previous)
    return -torch.mean(loss)  # Negative sign for gradient descent optimization

# Initialize the model, optimizer, and dataset as before
model = SimpleLLM(input_dim, output_dim)
opponent = SimpleLLM(input_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-1)
X = torch.randn(N, input_dim)
y = torch.zeros_like(torch.randint(0, 2, (N, output_dim)).float())

print("Pre", model(X).mean())
opponent.load_state_dict(model.state_dict())
# Training loop
for t in range(T):
    
    X = torch.randn(N, input_dim)
    # Generate synthetic data y' using the current model parameters
    y_synthetic = opponent(X).detach()
    
    # Update the model parameters
    optimizer.zero_grad()
    loss = spin_loss(model, opponent, X, y, y_synthetic)
    # Save the current weights for theta_t
    opponent.load_state_dict(model.state_dict())
    loss.backward()
    optimizer.step()
    
    if t % 10 == 0:
        print(f"Iteration {t}, Loss: {loss.item()}")

X = torch.randn(N, input_dim)
# Output the final model parameters
theta_T = model.linear.weight.data
print("Final model parameters (theta_T):", theta_T)
print("Result", model(X).mean())


