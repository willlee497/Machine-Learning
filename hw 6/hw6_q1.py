import torch
import torch.nn as nn
import torch.optim as optim


class LinearAE(nn.Module):
    def __init__(self, d_input: int, d_hidden: int):
        super().__init__()
        ### YOUR IMPLEMENTATION START ###
        #use a single weight matrix for tied weights (encoder transpose = decoder)
        #this ensures the autoencoder learns the same subspace as PCA
        self.weight = nn.Parameter(torch.randn(d_input, d_hidden) * 0.01)
        ### YOUR IMPLEMENTATION END ###

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ### YOUR IMPLEMENTATION START ###
        #encoder: z = x @ W (project input to latent space)
        return x @ self.weight
        ### YOUR IMPLEMENTATION END ###

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        ### YOUR IMPLEMENTATION START ###
        #decoder: x_hat = z @ W^T (reconstruct from latent space)
        return z @ self.weight.T
        ### YOUR IMPLEMENTATION END ###

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def autoencode(data: torch.Tensor):
    ### YOUR IMPLEMENTATION START ###
    # Train an linear autoencoder from the provided data
    # Return the encoded components
    
    #center the data
    data_mean = data.mean(dim=0, keepdim=True)
    centered_data = data - data_mean
    
    #get data dimensions and set latent dimension to 2 (for PCA comparison)
    n_samples, n_features = centered_data.shape
    latent_dim = 2
    
    #create the linear autoencoder model
    model = LinearAE(n_features, latent_dim)
    
    #set up training components with better parameters
    criterion = nn.MSELoss()  #reconstruction loss
    optimizer = optim.Adam(model.parameters(), lr=0.002)  #lower LR for final precision tuning
    
    #training loop
    num_epochs = 8000  #more epochs for precision
    prev_loss = float('inf')
    
    for epoch in range(num_epochs):
        #forward pass on centered data
        reconstructed = model(centered_data)
        loss = criterion(reconstructed, centered_data)
        
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #print progress occasionally
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
        
        #early stopping if converged, with much tighter convergence for precision
        if abs(prev_loss - loss.item()) < 1e-12:
            print(f'Converged at epoch {epoch+1}')
            break
        prev_loss = loss.item()
    
    #return the encoded components of centered data (2D representation)
    with torch.no_grad():
        ae_components = model.encode(centered_data)
    
    ### YOUR IMPLEMENTATION END ###
    return ae_components
