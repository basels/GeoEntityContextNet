import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from os.path import join
from tqdm import tqdm
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.losses import NTXentLoss
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import KFold, train_test_split

# ----------- globals -----------

log = logging.getLogger(__name__)
CUDA_DEVICE = 'cuda:0'


# ----------- transformations -----------


class MyRotationTransform:
    """Rotate by one of the given angles."""
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)


OSMImageTransformations = [transforms.functional.invert,
                           transforms.Resize((200, 200)),
                           transforms.Grayscale(num_output_channels=1),
                           transforms.RandomAdjustSharpness(sharpness_factor=0, p=1.0),
                           MyRotationTransform(angles=[0, 90, 180, 270]),
                           transforms.RandomHorizontalFlip(p=0.5),
                           transforms.RandomVerticalFlip(p=0.5),
                           transforms.functional.to_pil_image,
                           transforms.ToTensor()]
    
# ----------- training utils -----------


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader, set_name='val'):
    model.eval()
    outputs = [model.validation_step(batch, set_name) for batch in val_loader]
    return model.validation_epoch_end(outputs, set_name)


# ----------- model utils ----------


def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels_list = []

    with torch.no_grad():
        for i_x, labels, s_x, n_x in dataloader:
            i_x, s_x, n_x = i_x.to(device), s_x.to(device), n_x.to(device)
            embedding = model(i_x, s_x, n_x)
            embeddings.append(embedding.cpu())
            labels_list.append(labels)

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels_list, dim=0)

    return embeddings, labels


def apply_tsne(embeddings, n_components=2, perplexity=30, n_iter=1000):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
    return tsne.fit_transform(embeddings)


def training_contrastive_simple(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for i, (shape_features, labels, spatial_features, contextual_features) in tqdm(enumerate(dataloader), total=len(dataloader)):
        shape_features, spatial_features, contextual_features = shape_features.to(device), spatial_features.to(device), contextual_features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        embeddings = model(shape_features, spatial_features, contextual_features)
        loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


# ----------- loss -------------


class NTXentLossWithTaxonomy(NTXentLoss):
    def __init__(self, taxonomy_matrix, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.taxonomy_matrix = taxonomy_matrix

    def compute_weights(self, a2, n, labels):
        label_a2 = labels[a2]
        label_an = labels[n]

        weights = self.taxonomy_matrix[label_a2, label_an]
        return weights

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple, labels):
        a1, p, a2, n = indices_tuple
        weights = self.compute_weights(a2, n, labels)

        if len(a1) > 0 and len(a2) > 0:
            dtype = neg_pairs.dtype
            
            if not self.distance.is_inverted:
                pos_pairs = -pos_pairs
                neg_pairs = -neg_pairs
            
            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = (neg_pairs * weights) / self.temperature

            n_per_p = c_f.to_dtype(a2.unsqueeze(0) == a1.unsqueeze(1), dtype=dtype)
            neg_pairs = neg_pairs * n_per_p
            neg_pairs[n_per_p == 0] = c_f.neg_inf(dtype)

            max_val = torch.max(pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]).detach()
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)
            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator / denominator) + c_f.small_val(dtype))
            
            return {
                "loss": {
                    "losses": -log_exp,
                    "indices": (a1, p),
                    "reduction_type": "pos_pair",
                }
            }
        return self.zero_losses()


# ----------- datasets -----------


class OSMShapeSpaceNeighborhoodDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir

        if transform is None:
            transform=transforms.Compose( [transforms.functional.invert,
                                           transforms.Resize((200, 200)),
                                           transforms.Grayscale(num_output_channels=1),
                                           transforms.functional.to_pil_image,
                                           transforms.ToTensor()])
        
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        
        df_obj = self.dataframe.iloc[index]
        
        img_fname = df_obj.img
        img_path = join(self.img_dir, img_fname)
        image = read_image(img_path)
        
        if self.transform:
            image = self.transform(image)
        label = df_obj.target_label_enc
        
        # not used: shp_type, shp_centroid, shp_bounds, shp_minclearance
        features_spatial = torch.FloatTensor([df_obj.shp_area,
                                              df_obj.shp_length])
        
        features_neighborhood = torch.FloatTensor(df_obj.neigh_emb)

        return image, label, features_spatial, features_neighborhood
    
    
# ----------- models -----------


class OSMShapeSpatialNeighborhoodContrastive(nn.Module):
    def __init__(self, num_of_spatial_feats, num_of_context_feats):
        super().__init__()

        self.num_of_spatial_feats = num_of_spatial_feats
        self.num_of_context_feats = num_of_context_feats

        # Convolutional layers
        self.conv_1_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv_1_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.btchnorm1 = nn.BatchNorm2d(64)

        self.conv_2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.conv_2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.btchnorm2 = nn.BatchNorm2d(128)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=282752, out_features=128)
        self.btchnorm3 = nn.BatchNorm1d(128)

        fc2_num_features = 128 + self.num_of_spatial_feats + self.num_of_context_feats
        self.fc2 = nn.Linear(in_features=fc2_num_features, out_features=128)

        # Embedding Layer
        self.embedding_layer = nn.Linear(128, 300)
        
    def forward(self, i_x, s_x, n_x):
        x = i_x.float()

        # First convolutional block
        x = self.conv_1_1(x)
        x = F.leaky_relu(x)
        x = self.conv_1_2(x)
        x = self.btchnorm1(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # Second convolutional block
        x = self.conv_2_1(x)
        x = F.leaky_relu(x)
        x = self.conv_2_2(x)
        x = self.btchnorm2(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # Fully connected layers
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.btchnorm3(x)
        x = F.leaky_relu(x)

        # Concatenating shape, spatial, and contextual features
        x = torch.cat((x, s_x, n_x), axis=1)
        x = self.fc2(x)

        # Generating the final embedding
        embedding = self.embedding_layer(x)
        return embedding