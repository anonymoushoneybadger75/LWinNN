import torch
import torchvision.transforms.v2 as transforms
from src.General.Embedder import Embedder
from tqdm import tqdm
from src.LWinNN.LWinNN_Model import lwinnn_model

class lwinnn(torch.nn.Module):
    """
    lwinnn: backend class for LWinNN model. 
    The responsibility of this class is to combine functionality from Embedder class and LWinNN model class neatly. 
    It takes a dataloader and feeds the batches of images to the Embedder class. It then reshapes the embeddings in a format convenient for the model.

    Args:
        device (Torch device): device to write tensors to.
        backbone (str): name of the pretrained model. Default: 'resnet18'
        layers (tuple[str]): which layers to select from the backbone. Default: ['layer1', 'layer2', 'layer3']
        pool (bool): whether feature maps are pooled with a 1-strided average pooling operation before resizing. Default: True
        interpolation_mode (str): mode of interpolation used when resizing feature maps. Default: 'bilinear'
        window_size (int): size of local windows. Default: 5
        limit_train_samples (int): amount of samples to be selected for training. -1 if all samples need to be selected. Default: -1
    """
    def __init__(
            self,
            device=torch.device('cpu'),
            backbone='resnet18',
            layers=('layer1', 'layer2', 'layer3'),
            pool=True,
            interpolation_mode='bilinear',
            window_size=5,
            limit_train_samples=-1,
            ):
        super(lwinnn, self).__init__()
        self.device = device
        self.embedder = Embedder(backbone=backbone, layers=layers, pool=pool, interpolation_mode=interpolation_mode)
        self.limit_train_samples = limit_train_samples
        self.model = lwinnn_model(window_size=window_size)
        self.padder = transforms.Pad(padding=window_size//2, padding_mode='edge')
        torch.manual_seed(2024)


    """
    fit: function that takes a dataloader, generates embeddings for all images, and writes these embeddings to the memory bank.
    Args:
        dataloader (Torch dataloader): dataloader with batches of images.
    """
    def fit(self, dataloader):
        embeddings = []
        # if no limit on train samples, limit train samples with dataset size.
        if self.limit_train_samples < 0:
            self.limit_train_samples = len(dataloader.dataset)

        for _, data in tqdm(enumerate(dataloader), desc='Extracting embeddings'):
            data = data.to(self.device)[:self.limit_train_samples]
            embeddings.append(self.reshape_embedding(self.embedder(data), train=True))
            self.limit_train_samples -= data.shape[0]
            if self.limit_train_samples <= 0:
                break 
            self.empty_cache()

        embeddings = torch.cat(embeddings, dim=2)
        self.model.set_memory_bank(embeddings)
        del embeddings
        self.empty_cache()

    
    """
    predict: calculates anomaly scores for all batches in the dataloader.
    Args:
        dataloader (Torch dataloader): dataloader with batches of images.
    Returns:
        image_anomaly_scores (Torch tensor): tensor of shape N containing an anomaly score for every image.
        pixel_anomaly_scores (Torch tensor): tensor of shape N x 1 x H_0 x W_0 containing an anomaly score for every pixel.
    """
    def predict(self, dataloader):
        anomaly_scores = []
        for _, data in tqdm(enumerate(dataloader), desc='Calculating anomaly scores'):
            data = data.to(self.device)
            anomaly_scores.append(self.forward(data))
            self.empty_cache()
        image_anomaly_scores = torch.cat([anomaly_score[0] for anomaly_score in anomaly_scores], dim=0)
        pixel_anomaly_scores = torch.cat([anomaly_score[1] for anomaly_score in anomaly_scores], dim=0)
        return image_anomaly_scores, pixel_anomaly_scores
    

    """
    reshape_embedding: reshapes embedding to be in a more convenient shape. 
    The goal here is to move the height and width dimensions to the front to enable batched matric multiplication.
    Args:
        embedding (Torch tensor): embedding of shape B x C x H_1 x W_1
        train (bool): boolean indicating whether the embedding is generated during training, and therefore needs padding.
    Returns:
        (Torch tensor): reshaped embedding of shape H_1 x W_1 x B x C
    """
    def reshape_embedding(self, embedding, train=True):
        # pad sides for local window search
        if train:
            embedding = self.padder(embedding)
        # move height and width dimensions to front, and sample dimension to back. 
        return embedding.permute(2,3,0,1)
    

    """
    forward: calculates anomaly scores for a batch of images.
    Args:
        data (Torch tensor): images of shape B x 3 x H_0 x W_0
    Returns:
        (Torch tensor): image anomaly score of shape B
        (Torch tensor): pixel anomaly score of shape B x 1 x H_0 x W_0
    """
    def forward(self, data):
        embedding = self.reshape_embedding(self.embedder(data), train=False)
        return self.model.forward(embedding, output_size=data.shape)
    

    """
    empty_cache: empties cache. Exists to create compatibility for mps devices.
    """
    def empty_cache(self):
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        # Clearing mps cache seems to provide inconsistent results, unclear whether this is an implementation issue or a bug in torch mps support. 
        # elif "mps" in str(self.device):
        #     torch.mps.empty_cache()
