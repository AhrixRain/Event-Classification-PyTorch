import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import h5py
import seaborn as sns
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

############################################################################
# make_dataset
############################################################################

class Dataset_dikshant(Dataset):
    def __init__(self, filepath_list, transform=None):
        self.filepath_list = filepath_list
        self.length = len(filepath_list)
        self.transform = transform
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        target_dir = self.filepath_list[idx].split('/')[-1]
        # Load cropped scar image
        img_path = self.filepath_list[idx]
        image = [None, None]
        image[0] = Image.open(f'{img_path}/{target_dir}_xz.png')
        image[1] = Image.open(f'{img_path}/{target_dir}_yz.png')
        image0 = torch.tensor(np.expand_dims(image[0], axis=0))
        image1 = torch.tensor(np.expand_dims(image[1], axis=0))
        # load label array
        pid = None
        label = 0
        with open(f'{img_path}/{target_dir}_pid.txt', 'r', encoding='utf-8') as file:
            pid = file.read()
        if pid=='nuecc':
            label=0
        elif pid=='numucc':
            label=1
        else:
            label=2
        label = torch.tensor(label)      
        return image0, image1, label

def classifier_dataloader_cropped(batch_size, shuffle):
    """
    Build dataloader for classification with cropped images
    """
    # Set Image Transform (removed normalize for grayscale images)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Transform to tensor only
    ])

    class_paths = glob.glob("/home/ziqinl12/Desktop/asCroppedBlackOnWhiteImages_HTCONDOR_OUTPUT/*/*")
    train_paths, test_paths = train_test_split(class_paths, test_size=0.2, random_state=77, shuffle=True)
    train_dataset = Dataset_dikshant(train_paths, transform=transform)
    test_dataset = Dataset_dikshant(test_paths, transform=transform)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    print(f"Build dataset success. ")
    
    return train_loader, test_loader

############################################################################
# models
############################################################################

class DualImageResNet18Gray(nn.Module):
    def __init__(self, num_classes=3):
        super(DualImageResNet18Gray, self).__init__()
        
        # Load a pre-defined ResNet-18 model
        self.resnet18 = models.resnet18()
        
        # Modify the first convolutional layer to accept a single-channel input
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Split the ResNet into two parts: feature extractor and classifier
        self.adjust_channels = nn.Conv2d(128, 64, kernel_size=1)
        self.feature_extractor = nn.Sequential(
            self.resnet18.conv1,
            self.resnet18.bn1,
            self.resnet18.relu,
            self.resnet18.maxpool,
            self.resnet18.layer1,
            self.resnet18.layer2,
            self.adjust_channels
        )
        
        self.remaining_layers = nn.Sequential(
            self.resnet18.layer3,
            self.resnet18.layer4,
            self.resnet18.avgpool
        )
        
        # Modify the fully connected layer to match the number of classes
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, img1, img2):
        # Extract features from both images
        features1 = self.feature_extractor(img1)
        features2 = self.feature_extractor(img2)
        
        # Combine features (e.g., by concatenation or addition)
        combined_features = torch.cat((features1, features2), dim=1)  # Concatenate along channel dimension
        
        # Pass through remaining layers
        x = self.remaining_layers(combined_features)
        x = torch.flatten(x, 1)
        x = self.resnet18.fc(x)
        
        return x


class DualImageResNet34Gray(nn.Module):
    def __init__(self, num_classes=3):
        super(DualImageResNet34Gray, self).__init__()
        
        # Load a pre-defined ResNet-18 model
        self.resnet34 = models.resnet34()
        
        # Modify the first convolutional layer to accept a single-channel input
        self.resnet34.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Split the ResNet into two parts: feature extractor and classifier
        self.adjust_channels = nn.Conv2d(128, 64, kernel_size=1)
        self.feature_extractor = nn.Sequential(
            self.resnet34.conv1,
            self.resnet34.bn1,
            self.resnet34.relu,
            self.resnet34.maxpool,
            self.resnet34.layer1,
            self.resnet34.layer2,
            self.adjust_channels
        )
        
        self.remaining_layers = nn.Sequential(
            self.resnet34.layer3,
            self.resnet34.layer4,
            self.resnet34.avgpool
        )
        
        # Modify the fully connected layer to match the number of classes
        self.resnet34.fc = nn.Linear(self.resnet34.fc.in_features, num_classes)

    def forward(self, img1, img2):
        # Extract features from both images
        features1 = self.feature_extractor(img1)
        features2 = self.feature_extractor(img2)
        
        # Combine features (e.g., by concatenation or addition)
        combined_features = torch.cat((features1, features2), dim=1)  # Concatenate along channel dimension
        
        # Pass through remaining layers
        x = self.remaining_layers(combined_features)
        x = torch.flatten(x, 1)
        x = self.resnet34.fc(x)
        
        return x


class DualImageResNet50Gray(nn.Module):
    def __init__(self, num_classes=3):
        super(DualImageResNet50Gray, self).__init__()
        
        # Load a pre-defined ResNet-18 model
        self.resnet50 = models.resnet50()
        
        # Modify the first convolutional layer to accept a single-channel input
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Split the ResNet into two parts: feature extractor and classifier
        self.adjust_channels = nn.Conv2d(512, 256, kernel_size=1)
        self.feature_extractor = nn.Sequential(
            self.resnet50.conv1,
            self.resnet50.bn1,
            self.resnet50.relu,
            self.resnet50.maxpool,
            self.resnet50.layer1,
            self.resnet50.layer2,
            self.adjust_channels
        )
        
        self.remaining_layers = nn.Sequential(
            self.resnet50.layer3,
            self.resnet50.layer4,
            self.resnet50.avgpool
        )
        
        # Modify the fully connected layer to match the number of classes
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)

    def forward(self, img1, img2):
        # Extract features from both images
        features1 = self.feature_extractor(img1)
        features2 = self.feature_extractor(img2)
        
        # Combine features (e.g., by concatenation or addition)
        combined_features = torch.cat((features1, features2), dim=1)  # Concatenate along channel dimension
        
        # Pass through remaining layers
        x = self.remaining_layers(combined_features)
        x = torch.flatten(x, 1)
        x = self.resnet50.fc(x)
        
        return x


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)

        hidden_dim = in_channels * expansion_factor
        layers = []
        if expansion_factor != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)


class SubNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SubNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1, bias=False),  # 512->256
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, out_channels, kernel_size=3, stride=2, padding=1, bias=False),  # 256->128
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MobileNetV2Modified(nn.Module):
    def __init__(self, num_classes=3, width_mult=1.0):
        super(MobileNetV2Modified, self).__init__()

        # Sub-networks process single-channel inputs and their outputs are concatenated.
        # Ensure the concatenated channels match the width multiplier setting.
        base_out = max(1, int(16 * width_mult))
        self.subnet1 = SubNet(1, base_out)
        self.subnet2 = SubNet(1, base_out)

        self.cfgs = [
            # t, c, n, s (expansion, channels, layers, stride)
            (1, 8, 1, 2),    # 512->256，
            (2, 16, 1, 2),   # 256->128
            (3, 24, 2, 2),   # 128->64
            (4, 32, 2, 2),   # 64->32
            (4, 48, 2, 2),   # 32->16
            (6, 64, 1, 1),   # 16
        ]

        # After concatenation, input channels equal 2 * base_out
        input_channels = 2 * base_out
        self.last_channels = int(128 * width_mult)  # 512->128

        self.features = [nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=2, padding=1, bias=False),  # 立即下采样
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True)
        )]

        for t, c, n, s in self.cfgs:
            output_channels = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channels, output_channels, stride, expansion_factor=t))
                input_channels = output_channels

        self.features.append(nn.Sequential(
            nn.Conv2d(input_channels, self.last_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.last_channels),
            nn.ReLU6(inplace=True)
        ))

        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channels, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )

        self._initialize_weights()

    def forward(self, x1, x2):
        x1 = self.subnet1(x1)
        x2 = self.subnet2(x2)

        x = torch.cat([x1, x2], dim=1)

        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).reshape(x.size(0), -1)  
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


# class SubNet(nn.Module):
#     def __init__(self, in_channels, out_channels):

#         super(SubNet, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.net(x)


# class MobileNetV2Modified(nn.Module):
#     def __init__(self, num_classes=3, width_mult=1.0):

#         super(MobileNetV2Modified, self).__init__()

#         self.subnet1 = SubNet(1, 32)  
#         self.subnet2 = SubNet(1, 32) 

#         self.cfgs = [
#             (1, 16, 1, 1),
#             (6, 24, 2, 2),
#             (6, 32, 3, 2),
#             (6, 64, 4, 2),
#             (6, 96, 3, 1),
#             (6, 160, 3, 2),
#             (6, 320, 1, 1),
#         ]

#         input_channels = int(64 * width_mult) 
#         self.last_channels = int(1280 * width_mult)

#         self.features = [nn.Sequential(
#             nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(input_channels),
#             nn.ReLU6(inplace=True)
#         )]

#         for t, c, n, s in self.cfgs:
#             output_channels = int(c * width_mult)
#             for i in range(n):
#                 stride = s if i == 0 else 1
#                 self.features.append(InvertedResidual(input_channels, output_channels, stride, expansion_factor=t))
#                 input_channels = output_channels

#         self.features.append(nn.Sequential(
#             nn.Conv2d(input_channels, self.last_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(self.last_channels),
#             nn.ReLU6(inplace=True)
#         ))

#         self.features = nn.Sequential(*self.features)

#         self.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(self.last_channels, num_classes)
#         )

#         self._initialize_weights()

#     def forward(self, x1, x2):

#         x1 = self.subnet1(x1)
#         x2 = self.subnet2(x2)

#         x = torch.cat([x1, x2], dim=1)

#         x = self.features(x)
#         x = F.adaptive_avg_pool2d(x, 1).reshape(x.size(0), -1)  
#         x = self.classifier(x)
#         return x

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.zeros_(m.bias)


############################################################################
# plotting
############################################################################

def save_cm(epoch, labels_list, preds_list, output_dir, model_name):
    sklearn_cm = confusion_matrix(labels_list, preds_list)
    sklearn_cm = sklearn_cm[0:3, 0:3]
    # normalization
    row_sums = sklearn_cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 
    sklearn_cm_norm = sklearn_cm.astype('float') / row_sums
    # Display labels. 
    sklearn_disp = ConfusionMatrixDisplay(
        confusion_matrix=sklearn_cm_norm,
        display_labels=['nue', 'numu', 'nc']  
    )
    # Generate fig. 
    fig, ax = plt.subplots(figsize=(8, 6))
    sklearn_disp.plot(
        cmap=plt.cm.Blues,
        ax=ax,
        values_format='.2f',
        colorbar=False
    )
    # Adjust color and layout. 
    plt.colorbar(sklearn_disp.im_, ax=ax, fraction=0.046, pad=0.04)
    plt.title(f'Normalized Confusion Matrix - {model_name} (epoch={epoch+1})')
    # Adjust labels. 
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    # Close and save. 
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cm_{model_name}_epoch_{epoch+1}.png", dpi=300, bbox_inches='tight')
    plt.close()

def load_predictions(path):
    """Load prediction results from HDF5 file"""
    with h5py.File(path, 'r') as hf:
        probs = hf['probs'][:]
        labels = hf['labels'][:]
        files = hf['files'][:]
    return probs, labels, files

def infer_class_labels(num_cols, use_four=False):
    """Infer class label names based on number of classes"""
    if num_cols == 4 or use_four:
        return ['NueCC', 'NumuCC', 'NutauCC', 'NC']
    elif num_cols == 3:
        return ['NueCC', 'NumuCC', 'NC']
    else:
        return [f'C{i}' for i in range(num_cols)]

def sample_distribution_plot(labels, pred, class_names, out_dir, tag):
    """Sample distribution histogram"""
    plt.figure(figsize=(8, 6))
    plt.hist([labels, pred],
             bins=len(class_names),
             range=(0, len(class_names)),
             histtype='step',
             color=['blue', 'red'],
             label=['True', 'Predicted'])
    plt.xlabel('Label')
    plt.ylabel('Events')
    plt.legend(loc='upper right')
    
    plt.gca().set_xticks(np.arange(len(class_names)) + 0.5)
    plt.gca().set_xticklabels(class_names, ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/a_sample_distribution_{tag}.png', dpi=150)
    plt.close()

def class_pid_plots(probs, labels, class_names, out_dir, tag):
    """Generate PID threshold distribution plots and FOM/Eff/Pur curves"""
    import numpy as np
    n_classes = len(class_names)
    metrics = {}
    
    for class_idx in range(n_classes):
        class_name = class_names[class_idx]
        class_scores = probs[:, class_idx]
        
        # Build histogram data
        nbins = 100
        bins = np.linspace(0, 1, nbins+1)
        class_hist_data = []
        for true_class in range(n_classes):
            hist, _ = np.histogram(class_scores[labels == true_class], bins=bins)
            class_hist_data.append(hist)
        class_hist_data = np.array(class_hist_data)
        
        # Create subplot
        f, ax = plt.subplots(2, 2, figsize=(12, 10))
        f.subplots_adjust(hspace=0.3, wspace=0.3)
        
        colors = ['blue', 'pink', 'purple', 'red'][:n_classes]
        
        # Score distribution plot
        for i in range(n_classes):
            hist_data = class_hist_data[i]
            ax[0, 0].step(bins[:-1], hist_data, where='post', 
                         color=colors[i], label=f'True {class_names[i]}')
        
        ax[0, 0].set_ylabel('Events')
        ax[0, 0].set_xlabel(f'{class_name} Score')
        ax[0, 0].set_title('Score Distribution')
        ax[0, 0].legend(loc='upper right', fontsize=8)
        ax[0, 0].grid(True, alpha=0.3)
        
        # Calculate metrics
        target_sel = class_hist_data[class_idx][::-1].cumsum()[::-1]
        total_sel = np.sum(class_hist_data, axis=0)[::-1].cumsum()[::-1]
        
        class_total = max(class_hist_data[class_idx].sum(), 1)
        eff = target_sel / class_total
        
        pur = np.zeros_like(total_sel, dtype=float)
        nonzero_mask = total_sel > 0
        pur[nonzero_mask] = target_sel[nonzero_mask] / total_sel[nonzero_mask]
        
        fom = eff * pur
        
        # Find optimal threshold
        min_bin, max_bin = 5, 95
        valid_range = slice(min_bin, max_bin)
        
        if np.sum(fom[valid_range]) > 0 and np.max(fom[valid_range]) > 0:
            best_bin = min_bin + np.argmax(fom[valid_range])
        else:
            best_bin = 50
        
        best_thr = bins[best_bin]
        
        # Efficiency/Purity/FOM curves
        ax[0, 1].step(bins[:-1], eff, where='post', color='red', label='Efficiency')
        ax[0, 1].step(bins[:-1], pur, where='post', color='blue', label='Purity')
        ax[0, 1].step(bins[:-1], fom, where='post', color='green', label='FOM')
        ax[0, 1].axvline(x=best_thr, color='gray', linestyle='--', alpha=0.7)
        ax[0, 1].set_ylabel('Value')
        ax[0, 1].set_xlabel(f'{class_name} Score')
        ax[0, 1].set_title('Efficiency, Purity & FOM')
        ax[0, 1].legend(loc='lower center', fontsize=8)
        ax[0, 1].grid(True, alpha=0.3)
        
        # ROC curve
        y_true_binary = (labels == class_idx).astype(int)
        y_scores = class_scores
        
        try:
            fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
            roc_auc = auc(fpr, tpr)
            
            ax[1, 0].plot(fpr, tpr, color='darkorange', lw=2,
                         label=f'{class_name} (AUC = {roc_auc:.3f})')
            ax[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                         label='Random (AUC = 0.500)')
            ax[1, 0].set_xlabel('False Positive Rate')
            ax[1, 0].set_ylabel('True Positive Rate')
            ax[1, 0].set_title(f'ROC Curve - {class_name}')
            ax[1, 0].legend(loc="lower right", fontsize=8)
            ax[1, 0].grid(True, alpha=0.3)
        except:
            roc_auc = 0.0
            ax[1, 0].text(0.5, 0.5, 'ROC calculation failed', 
                         ha='center', va='center', transform=ax[1, 0].transAxes)
        
        # Precision-Recall curve
        try:
            precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
            avg_precision = average_precision_score(y_true_binary, y_scores)
            
            ax[1, 1].plot(recall, precision, color='darkorange', lw=2,
                         label=f'{class_name} (AP = {avg_precision:.3f})')
            baseline = np.sum(y_true_binary) / len(y_true_binary)
            ax[1, 1].axhline(y=baseline, color='navy', linestyle='--', lw=2,
                           label=f'Baseline (AP = {baseline:.3f})')
            ax[1, 1].set_xlabel('Recall')
            ax[1, 1].set_ylabel('Precision')
            ax[1, 1].set_title(f'Precision-Recall Curve - {class_name}')
            ax[1, 1].legend(loc="lower left", fontsize=8)
            ax[1, 1].grid(True, alpha=0.3)
        except:
            avg_precision = 0.0
            ax[1, 1].text(0.5, 0.5, 'PR curve calculation failed', 
                         ha='center', va='center', transform=ax[1, 1].transAxes)
        
        f.suptitle(f'{class_name} Classification Analysis', fontsize=14, y=0.95)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/{class_name.lower()}_analysis_{tag}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Store metrics
        class_sel = {}
        class_eff = {}
        class_pur = {}
        
        for i in range(n_classes):
            sel_i = class_hist_data[i][::-1].cumsum()[::-1]
            class_sel[class_names[i]] = sel_i[best_bin]
            class_eff[class_names[i]] = sel_i[best_bin] / max(class_hist_data[i].sum(), 1)
            if total_sel[best_bin] > 0:
                class_pur[class_names[i]] = sel_i[best_bin] / total_sel[best_bin]
            else:
                class_pur[class_names[i]] = 0.0
        
        metrics[class_name] = {
            'threshold': best_thr,
            'eff': eff[best_bin],
            'pur': pur[best_bin],
            'fom': fom[best_bin],
            'auc': roc_auc,
            'avg_precision': avg_precision,
            'class_eff': class_eff,
            'class_pur': class_pur,
            'class_sel': class_sel
        }
    
    return metrics

def plot_confusion_matrix(matrix, class_names, output_path, title):
    """Plot confusion matrix"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    # Check data type
    is_count_data = np.all(matrix >= 1) and np.any(matrix > 10)
    
    if is_count_data:
        cax = ax.matshow(matrix, cmap='Blues')
    else:
        cax = ax.matshow(matrix, cmap='Blues', vmin=0.0, vmax=1.0)
    
    fig.colorbar(cax)
    plt.title(title, y=1.08, fontsize=14)
    
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    plt.xlabel('Predicted')
    ax.xaxis.set_label_position('top') 
    plt.ylabel('True')
    
    # Add values in cells
    for (i, j), z in np.ndenumerate(matrix):
        if abs(z) < 1e-6:
            continue
        
        if is_count_data:
            text = f'{int(z)}'
            threshold = np.max(matrix) * 0.5
        else:
            text = f'{z:.3f}'
            threshold = 0.3
        
        text_color = 'white' if z > threshold else 'black'
        weight = 'bold' if z > threshold else 'normal'
        ax.text(j, i, text, ha='center', va='center', 
                color=text_color, fontweight=weight, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_calculated_matrices(class_names, out_dir, tag, trueid, pred):
    """Plot calculated efficiency and purity matrices"""
    from sklearn.metrics import confusion_matrix
    n_classes = len(class_names)
    
    conf_matrix = confusion_matrix(trueid, pred, labels=range(n_classes))
    
    # Efficiency matrix
    eff_matrix = np.zeros((n_classes, n_classes), dtype=float)
    row_sums = conf_matrix.sum(axis=1)
    for i in range(n_classes):
        if row_sums[i] > 0:
            eff_matrix[i] = conf_matrix[i] / row_sums[i]
    
    # Purity matrix
    pur_matrix = np.zeros((n_classes, n_classes), dtype=float)
    col_sums = conf_matrix.sum(axis=0)
    for j in range(n_classes):
        if col_sums[j] > 0:
            pur_matrix[:, j] = conf_matrix[:, j] / col_sums[j]
    
    plot_confusion_matrix(eff_matrix, class_names, 
                         f'{out_dir}/calc_eff_{tag}.png',
                         'Calculated Efficiency')
    
    plot_confusion_matrix(pur_matrix, class_names, 
                         f'{out_dir}/calc_pur_{tag}.png',
                         'Calculated Purity')
