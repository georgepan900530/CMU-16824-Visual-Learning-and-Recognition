import torch
import torch.utils
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import torchvision
import torch.nn as nn
import random
from sklearn.manifold import TSNE
import utils
import colorsys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        ##################################################################
        # TODO: Define a FC layer here to process the features
        ##################################################################

        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze layer 4
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        # fc
        self.resnet.fc = nn.Linear(512, num_classes)

        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def forward(self, x):
        ##################################################################
        # TODO: Return raw outputs here
        ##################################################################
        out = self.resnet(x)
        return out
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


# Get features for t-SNE
def get_features(model, test_loader, args):
    model.eval()
    labels = []
    all_features = []
    with torch.no_grad():
        for img, label, _ in test_loader:
            img = img.to(args.device)
            label = label.to(args.device)
            label = label.cpu().numpy()
            labels.append(label)
            output = model(img)
            output = torch.flatten(output, 1)
            output = output.cpu().numpy()
            all_features.append(output)

    return labels, all_features


# We need a function to compute the mean color for instances with multiple labels
def compute_mean_color(labels, colors_mapping, idx2class):
    present_classes = [idx2class[i] for i in range(20) if labels[i] == 1]
    # return black if no class is present
    if len(present_classes) == 0:
        return np.array([0, 0, 0])

    # Get colors for all present classes
    colors = np.array([colors_mapping[name] for name in present_classes])
    mean_color = np.mean(colors, axis=0)

    return mean_color / 255.0  # Accomodate for matplotlib


def normalize_color(color):
    return [c / 255.0 for c in color]


args = ARGS(
    epochs=50,
    inp_size=224,  # originally 64
    use_cuda=True,
    val_every=70,
    lr=0.001,
    batch_size=128,
    step_size=10,
    gamma=0.1,
    save_freq=10,
    save_at_end=True,
    weight_decay=1e-4,
    smoothing_factor=0.3,
    test_batch_size=100,
)

class_colors_rgb = [
    [230, 25, 75],  # Red
    [60, 180, 75],  # Green
    [255, 225, 25],  # Yellow
    [0, 130, 200],  # Blue
    [245, 130, 48],  # Orange
    [145, 30, 180],  # Purple
    [70, 240, 240],  # Cyan
    [240, 50, 230],  # Magenta
    [210, 245, 60],  # Lime
    [250, 190, 212],  # Pink
    [0, 128, 128],  # Teal
    [220, 190, 255],  # Lavender
    [170, 110, 40],  # Brown
    [255, 250, 200],  # Beige
    [128, 0, 0],  # Maroon
    [170, 255, 195],  # Mint
    [128, 128, 0],  # Olive
    [255, 215, 180],  # Coral
    [0, 0, 128],  # Navy
    [128, 128, 128],  # Grey
]

colors_mapping = {}
class_names = VOCDataset.CLASS_NAMES
class2idx = VOCDataset.INV_CLASS
idx2class = {i: name for name, i in class2idx.items()}
for i in range(len(class_names)):
    colors_mapping[class_names[i]] = class_colors_rgb[i]

model = ResNet(len(VOCDataset.CLASS_NAMES))
model.load_state_dict(
    torch.load("checkpoints/q2_resnet_best/checkpoint-model-epoch50.pth")
)
model = model.to(args.device)
test_dataset = VOCDataset(split="test", size=args.inp_size)
random_sampler = torch.utils.data.RandomSampler(test_dataset, num_samples=1000)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, sampler=random_sampler
)

labels, all_features = get_features(model, test_loader, args)
labels = np.vstack(labels)
all_features = np.vstack(all_features)
tsne = TSNE(
    n_components=2, random_state=1124, init="pca", perplexity=40, learning_rate=500
)
features_2d = tsne.fit_transform(all_features)

# calculate mean color for each image
colors = []
for label in labels:
    mean_color = compute_mean_color(label, colors_mapping, idx2class)
    colors.append(mean_color)

tx, ty = features_2d[:, 0], features_2d[:, 1]


legend_elements = [
    Line2D(
        [0],
        [0],
        color=normalize_color(colors_mapping[class_name]),
        lw=4,
        label=f"{class_name}",
    )
    for class_name in colors_mapping
]

colors = np.array(colors)
plt.figure(figsize=(10, 10))
plt.scatter(tx, ty, c=colors)
plt.title("t-SNE of ResNet features")
plt.legend(handles=legend_elements, loc="upper right")
plt.savefig("tsne.png")
