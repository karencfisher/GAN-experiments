import torch
from torch.utils.data import DataLoader, Dataset
from time import time


def timer(func):
    # Decorator to time a function
    def wrapper_func(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        elapsed_time = time() - start_time
        hours = int(elapsed_time // 3600)
        elapsed_time %= 3600
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        print(f'{func.__name__} - time taken: {hours:02d}:{minutes:02d}:{seconds:06.3f}')
        return result
    return wrapper_func

class Eigenloss:
    def __init__(self, n_components: int = 100):
        self.n_components = n_components
        self.computed_mean = None
        self.eigenvectors = None
        
    @timer
    def train(self, data: torch.Tensor, method: str = 'svd'):
        """
        Train eigenloss model. By default this uses an SVD-based approach which avoids
        forming the full covariance matrix and runs on GPU if available.

        method: 'auto'|'svd'|'eigh'
        """
        # Decide device: prefer CUDA if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)

        # Compute the mean on the working device
        self.computed_mean = torch.mean(data, dim=0)
        # Center the data
        centered_data = data - self.computed_mean

        # If method is auto, pick a heuristic based on data shape. Otherwise use explicit method.
        if method == 'auto':
            N, D = centered_data.shape
            method = 'eigh' if D <= min(2000, max(1, N // 2)) else 'svd'

        # Use SVD since it's often faster and more memory-efficient for large D
        if method == 'svd':
            # Use SVD on the centered data: X = U S Vh -> columns of V are eigenvectors of X^T X
            # This avoids explicitly forming the (D x D) covariance when D is large.
            # Ensure float32 for GPU linear algebra
            centered_data = centered_data.to(torch.float32)
            # Compute economy SVD
            # For input X shape (N, D), torch.linalg.svd returns U (N, k), S (k), Vh (k, D) where k = min(N, D)
            U, S, Vh = torch.linalg.svd(centered_data, full_matrices=False)
            V = Vh.transpose(-2, -1)
            # V has shape (D, k); take leading components
            k_available = V.shape[1]
            k = min(self.n_components, k_available)
            self.eigenvectors = V[:, :k].contiguous()
            return

        # Fallback to forming covariance and using eigh (may be faster when D small)
        data = data.to(torch.float32)
        covariance_matrix = torch.matmul(centered_data.T, centered_data) / (data.size(0) - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        self.eigenvectors = eigenvectors[:, sorted_indices[:self.n_components]]
    
    def compute_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.computed_mean is None or self.eigenvectors is None:
            raise ValueError("Eigenloss model has not been trained yet.")
        
        # flatten inputs from [batch_size, 3, 64, 64] to [batch_size, 12288]
        inputs = inputs.view(inputs.size(0), -1)
        # Move to same device as eigenvectors/mean
        device = self.eigenvectors.device
        inputs = inputs.to(device)
        # Center the inputs
        centered_inputs = inputs - self.computed_mean
        # Project onto eigenvectors
        projections = torch.matmul(centered_inputs, self.eigenvectors)
        # Reconstruct the inputs
        reconstructed = torch.matmul(projections, self.eigenvectors.T) + self.computed_mean
        # Compute reconstruction error
        loss = self._cosine_distance(inputs, reconstructed)
        return loss
    
    def _cosine_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_norm = a / a.norm(dim=1, keepdim=True)
        b_norm = b / b.norm(dim=1, keepdim=True)
        return 1 - torch.sum(a_norm * b_norm, dim=1).mean()
    
    
def get_eigenloss(data_loader: DataLoader, n_components: int = 100, n_batches: int = 100) -> Eigenloss:
    eigenloss = Eigenloss(n_components=n_components)
    all_data = []
    for i, batch in enumerate(data_loader):
        if i >= n_batches:
            break
        all_data.append(batch.view(batch.size(0), -1))
    all_data_tensor = torch.cat(all_data, dim=0)
    eigenloss.train(all_data_tensor)
    return eigenloss


if __name__ == "__main__":
    # unit test, make one batch of single color images
    dummy_data = torch.ones((10, 3, 64, 64)) * 0.5  # gray images
    dummy_loader = DataLoader(dummy_data, batch_size=5)
    eigenloss = get_eigenloss(dummy_loader, n_components=10, n_batches=2)
    loss = eigenloss.compute_loss(dummy_data)
    print(f"Eigenloss on dummy data: {loss.item()}")
    assert loss.item() < 1e-5, "Eigenloss should be very low for identical images."
    print("Eigenloss unit test passed.")
    
    
    
    