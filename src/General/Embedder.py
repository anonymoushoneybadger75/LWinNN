import torch
import torchvision
import torchvision.models.feature_extraction

class Embedder(torch.nn.Module):
    """
    Embedder: class that extracts features from batches of images with a pretrained model and forms the feature maps into an embedding.
    The methods here are fairly generic for IAD: many methods use similar approaches, therefore this file is not part of the LWinNN folder.
    Args:
        backbone (str): name of the pretrained model. Default: 'resnet18'
        layers (tuple[str]): which layers to select from the backbone. Default: ('layer1', 'layer2', 'layer3')
        pool (bool): whether feature maps are pooled with a 1-strided average pooling operation before resizing. Default: True
        interpolation_mode (str): mode of interpolation used when resizing feature maps. Default: 'bilinear'
    """
    def __init__(self,backbone='resnet18', layers=("layer1", "layer2", "layer3"), pool=False, interpolation_mode='bilinear'):
        super(Embedder, self).__init__()
        self.backbone = backbone
        self.interpolation_mode=interpolation_mode

        self.layers=list(layers)
        self.features_layers = {}
        for layer in self.layers:
            self.features_layers[layer] = 'feat'+layer[-1]

        models = {'resnet18': torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
                 'wide_resnet50': torchvision.models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1),
                 'wide_resnet101': torchvision.models.wide_resnet101_2(weights=torchvision.models.Wide_ResNet101_2_Weights.IMAGENET1K_V1),
                 'resnet34': torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT),
                 'resnet50': torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)}

        self.feature_extractor = models[backbone]
        del models
        self.feature_extractor = torchvision.models.feature_extraction.create_feature_extractor(self.feature_extractor, self.features_layers)
        self.feature_extractor.eval()

        self.pool=pool
        # padding can leave 0-artifacts on the border of the feature maps. 
        self.feature_pooler = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=0)


    """
    forward: function the generates an embedding from a batch of images
    Args: 
        data (Torch tensor): a tensor of shape B x 3 x H_0 x W_0 containing B images of shape H_0 x W_0
    Returns:
        embeddings (Torch tensor): a tensor of shape B x C x H_1 x W_1 containing embeddings.
    """
    def forward(self, data):
        with torch.no_grad():
            features = self.extract_features(data)
            embeddings = self.form_embeddings(features) 
            return embeddings
    

    """
    extract_features: function that extracts features from a batch of images with the backbone.
    Args: 
        data (Torch tensor): a tensor of shape B x 3 x H_0 x W_0 containing B images of shape H_0 x W_0.
    Returns 
        features (list[Torch tensor]): a list of tensors, having shape B x C_i x H_i x W_i for every i>0.
    """
    def extract_features(self, data):
        features = self.feature_extractor(data)
        if self.pool:
            features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        return features


    """
    form_embeddings: function that takes feature maps, resizes them to size H_1 x W_1, and concatenates them.
    Args:
        features (list[Torch tensor]): a list of tensors, having shape B x C_i x H_i x W_i for every i>0.
    Returns:
        (Torch tensor): a tensor of shape B x C x H_1 x W_1, with C the sum of all C_i. 
    """
    def form_embeddings(self, features):
        # append the first feature map into the list and store its shape
        embeddings = [features[self.features_layers[self.layers[0]]]]
        embedding_shape = embeddings[0].shape[-2:]

        # iterate over all other feature maps and resize them to the size of the first
        for layer in self.layers[1:]:
            embeddings.append(torch.nn.functional.interpolate(features[self.features_layers[layer]], size=embedding_shape, mode=self.interpolation_mode))
        # concatenate the resized feature maps on the channel dimension
        return torch.cat(embeddings, 1)
