import torch
import torchvision.transforms.v2 as transforms

class lwinnn_model(torch.nn.Module):
    """lwinnn_model: class that implements local window nearest neighbor.
    This class takes a batch of train embeddings, and writes them to memory. 
    For a test batch of embeddings this class uses local window nearest neighbors to calculate pixel and image anomaly scores.
    Args:
        window_size (int): size of local windows. Default: 5"""
    def __init__(self,window_size=5,):
        super(lwinnn_model, self).__init__()
        self.window_size = window_size
        
        # Gaussian blur to smooth anomaly maps.
        sigma = 4.0
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = transforms.GaussianBlur(kernel_size, sigma=sigma)


    """forward: function that calculates anomaly scores for a batch of embeddings.
    Args:
        embeddings (Torch tensor): batch of embeddings shape H_1 x W_1 x B x C
        output_size tuple(int, int): shape of original images (H_0,W_0)
    Returns:
        image_anomaly_scores (Torch tensor): image anomaly scores of shape B
        pixel_anomaly_scores (Torch tensor): pixel anomaly scores of shape B x 1 x H_0 x W_0
    """
    def forward(self, embeddings, output_size):
        with torch.no_grad():
            pixel_anomaly_scores = self.compute_pixel_anomaly_scores(embeddings)
            image_anomaly_scores = pixel_anomaly_scores.amax(dim=(1,2,3))
            pixel_anomaly_scores = torch.nn.functional.interpolate(pixel_anomaly_scores, size=output_size[-2:], mode='bilinear')
            return image_anomaly_scores.cpu(), self.blur(pixel_anomaly_scores).cpu()
    

    """set_memory_bank: setter function that writes train embeddings to memory.
    Args:
        embeddings (Torch tensor): train embeddings of shape H_1 x W_1 x N x C
    """
    def set_memory_bank(self, embeddings):
        self.memory_bank = embeddings


    """euclidean_dist: function to calculate batchwise euclidean dist. marginally faster than torch.cdist
    Args:
        x (Torch tensor): tensor of shape B_1 x B_2 x ... x M x C
        y (Torch tensor): tensor of shape B_1 x B_2 x ... x N x C
    Returns:
        (Torch tensor): batchwise euclidean distances between x and y, shape B_1 x B_2 x ... x M x N
    """
    def euclidean_dist(self, x, y):
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)  # |x|
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)  # |y|
        # row distance can be rewritten as sqrt(|x| - 2 * x @ y.T + |y|.T)
        res = x_norm - 2 * (x @ y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
        return res.clamp_min_(0).sqrt_()


    """nearest_neighbors: function that calculates the one-nearest neighbor between two batches of tensors.
    Args:
        test_embeddings (Torch tensor): tensor of shape H_1 x W_1 x B x C
        local_bank (Torch tensor): tensor of shape H_1 x W_1 x N_train x C
    Returns:
        (Torch tensor): euclidean distances to 1NN, shape H_1 x W_1 x B
    """
    def nearest_neighbors(self, test_embeddings, local_bank):
        # Batchwise euclidean distances have shape H_1 x W_1 x B x N_train. 
        distances = self.euclidean_dist(test_embeddings, local_bank)
        return distances.amin(dim=-1)


    """compute_pixel_scores: function that finds the nearest neighbor of every test embedding vector. 
    Rather than move a sliding window, we move our padded train embeddings around our test embedding by slicing over our window. 
    This is a suboptimal implementation but prevents out of memory exceptions in many cases.
    Args:
        test_embeddings (Torch tensor): test embeddings of shape H_1 x W_1 x B x C
    Returns:
        pixel_anomaly_scores (Torch tensor): pixel anomaly scores. shape B x 1 x H_1 x W_1
    """
    def compute_pixel_anomaly_scores(self, test_embeddings):
        H, W, _, _ = test_embeddings.shape
        pixel_anomaly_scores = None
        # iterate over all window_locations, and compare test embedding to a different slice of the memory bank. 
        for delta_1 in range(self.window_size):
            for delta_2 in range(self.window_size):
                # calculate 1NN for current slice
                current_neighbors = self.nearest_neighbors(test_embeddings, self.memory_bank[delta_1:H+delta_1, delta_2:W+delta_2, :, :])
                # set new anomaly scores to minimum between current old neighbors and current neighbors
                if pixel_anomaly_scores == None:
                    pixel_anomaly_scores = current_neighbors
                else:
                    pixel_anomaly_scores = torch.minimum(pixel_anomaly_scores, current_neighbors)

        # move batch dimension back to front and unsqueeze channel dimension.
        return pixel_anomaly_scores.permute(2,0,1)[:,None,:,:]
    