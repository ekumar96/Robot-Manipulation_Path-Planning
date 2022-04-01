import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
import image
import numpy as np
from random import seed
from segmentation_helper import check_dataloader, check_dataset
import matplotlib.pyplot as plt
from sim import get_tableau_palette
import torch.nn.functional as F

# ==================================================
mean_rgb = [0.485, 0.456, 0.406]
std_rgb = [0.229, 0.224, 0.225]
# ==================================================

class RGBDataset(Dataset):
    def __init__(self, img_dir):
        """
            Initialize instance variables.
            :param img_dir (str): path of train or test folder.
            :return None:
        """
        # TODO: complete this method
        # ===============================================================================
        # This is an instance of torch.utils dataset class, which has a lot of other member variables and functions
        # one of these will call __getitem__ in a for loop depending on dataset_length, and fill the dataset with samples

        # Input normalization info to be used in transforms.Normalize()
        mean_rgb = [0.722, 0.751, 0.807]
        std_rgb = [0.171, 0.179, 0.197]

        self.dataset_dir = img_dir

        self.has_gt = True

        # Transform to be applied on a sample.
        #  For this compose transforms.ToTensor() and transforms.Normalize() for RGB image should be enough.
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_rgb, std_rgb)])

        # Number of samples in the dataset.
        self.dataset_length = len(os.listdir(self.dataset_dir+"rgb"))
        # ===============================================================================

    def __len__(self):
        """
            Return the length of the dataset.
            :return dataset_length (int): length of the dataset, i.e. number of samples in the dataset
        """
        # TODO: complete this method
        # ===============================================================================
        return self.dataset_length
        # ===============================================================================

    def __getitem__(self, idx):
        """
            Given an index, return paired rgb image and ground truth mask as a sample.
            :param idx (int): index of each sample, in range(0, dataset_length)
            :return sample: a dictionary that stores paired rgb image and corresponding ground truth mask.
        """
        # TODO: complete this method
        # Hint:
        # - Use image.read_rgb() and image.read_mask() to read the images.
        # - Think about how to associate idx with the file name of images.
        # - Remember to apply transform on the sample.
        # ===============================================================================
        rgb_img = None
        gt_mask = None

        rgb_img_str = self.dataset_dir + "rgb/" + str(idx) + "_rgb.png"
        gt_mask_str = self.dataset_dir + "gt/" + str(idx) + "_gt.png"

        rgb_img = self.transform(image.read_rgb(rgb_img_str))

        gt_mask = torch.LongTensor(image.read_mask(gt_mask_str))
        sample = {'input': rgb_img, 'target': gt_mask}
        return sample
        # ===============================================================================


class miniUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super(miniUNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # TODO: complete this method
        # ===============================================================================
        
        self.conv1 = nn.Conv2d(in_channels = n_channels, out_channels = 16, kernel_size = 3, padding=True)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=True)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=True)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=True)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=True)
        self.conv6 = nn.Conv2d(256+128, 128, 3, padding=True)
        self.conv7 = nn.Conv2d(128+64, 64, 3, padding=True)
        self.conv8 = nn.Conv2d(64+32, 32, 3, padding=True)
        self.conv9 = nn.Conv2d(32+16, 16, 3, padding=True)
        self.conv10 = nn.Conv2d(16, n_classes, 1)
        # ===============================================================================

    def forward(self, x):
        # TODO: complete this method
        # ===============================================================================
        #print("\n\n")
        #print(x)
        #print(x.shape)
        #print(type(x))
        #print("\n\n")
        concat1 = F.relu(self.conv1(x))
        x = F.max_pool2d(concat1, (2, 2))

        concat2 = F.relu(self.conv2(x))
        x = F.max_pool2d(concat2, (2, 2))

        concat3 = F.relu(self.conv3(x))
        x = F.max_pool2d(concat3, (2, 2))

        concat4 = F.relu(self.conv4(x))
        x = F.max_pool2d(concat4, (2, 2))

        x = F.relu(self.conv5(x))

        x = F.interpolate(x, scale_factor=2)
        x = torch.cat((concat4, x), 1)
        x = F.relu(self.conv6(x))

        x = F.interpolate(x, scale_factor=2)
        x = torch.cat((concat3, x), 1)
        x = F.relu(self.conv7(x))

        x = F.interpolate(x, scale_factor=2)
        x = torch.cat((concat2, x), 1)
        x = F.relu(self.conv8(x))

        x = F.interpolate(x, scale_factor=2)
        x = torch.cat((concat1, x), 1)
        x = F.relu(self.conv9(x))

        x = self.conv10(x)

        output = x
        return output
        # ===============================================================================


def save_chkpt(model, epoch, test_miou, chkpt_path):
    """
        Save the trained model.
        :param model (torch.nn.module object): miniUNet object in this homework, trained model.
        :param epoch (int): current epoch number.
        :param test_miou (float): miou of the test set.
        :return: None
    """
    state = {'model_state_dict': model.state_dict(),
             'epoch': epoch,
             'model_miou': test_miou, }
    torch.save(state, chkpt_path)
    print("checkpoint saved at epoch", epoch)


def load_chkpt(model, chkpt_path, device):
    """
        Load model parameters from saved checkpoint.
        :param model (torch.nn.module object): miniUNet model to accept the saved parameters.
        :param chkpt_path (str): path of the checkpoint to be loaded.
        :return model (torch.nn.module object): miniUNet model with its parameters loaded from the checkpoint.
        :return epoch (int): epoch at which the checkpoint is saved.
        :return model_miou (float): miou of the test set at the checkpoint.
    """
    checkpoint = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    model_miou = checkpoint['model_miou']
    print("epoch, model_miou:", epoch, model_miou)
    return model, epoch, model_miou


def save_prediction(model, dataloader, dump_dir, device, BATCH_SIZE):
    """
        For all datapoints d in dataloader, save  ground truth segmentation mask (as {id}.png)
          and predicted segmentation mask (as {id}_pred.png) in dump_dir.
        :param model (torch.nn.module object): trained miniUNet model
        :param dataloader (torch.utils.data.DataLoader object): dataloader to use for getting predictions
        :param dump_dir (str): dir path for dumping predictions
        :param device (torch.device object): pytorch cpu/gpu device object
        :param BATCH_SIZE (int): batch size of dataloader
        :return: None
    """
    print(f"Saving predictions in directory {dump_dir}")
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
        
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_ID, sample_batched in enumerate(dataloader):
            data, target = sample_batched['input'].to(device), sample_batched['target'].to(device)
            output = model(data)
            _, pred = torch.max(output, dim=1)
            for i in range(pred.shape[0]):
                gt_image = convert_seg_split_into_color_image(target[i].cpu().numpy())
                pred_image = convert_seg_split_into_color_image(pred[i].cpu().numpy())
                combined_image = np.concatenate((gt_image, pred_image), axis=1)
                test_ID = batch_ID * BATCH_SIZE + i
                image.write_mask(combined_image, f"{dump_dir}/{test_ID}_gt_pred.png")


def iou(pred, target, n_classes=6):
    """
        Compute IoU on each object class and return as a list.
        :param pred (np.array object): predicted mask
        :param target (np.array object): ground truth mask
        :param n_classes (int): number of classes
        :return cls_ious (list()): a list of IoU on each object class
    """
    """
    prediction = pred
    _, pred = torch.max(prediction, dim=1)
    batch_num = prediction.shape[0]
    class_num = prediction.shape[1]
    batch_ious = list()
    for batch_id in range(batch_num):
        class_ious = list()
        for class_id in range(1, class_num):  # class 0 is background
            mask_pred = (pred[batch_id] == class_id).int()
            mask_target = (target[batch_id] == class_id).int()
            if mask_target.sum() == 0: # skip the occluded object
                continue
            intersection = (mask_pred * mask_target).sum()
            union = (mask_pred + mask_target).sum() - intersection
            class_ious.append(float(intersection) / float(union))
        batch_ious.append(np.mean(class_ious))
    return batch_ious
    """
    cls_ious = []
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(1, n_classes):  # class 0 is background
        pred_P = pred == cls
        target_P = target == cls
        pred_N = ~pred_P
        target_N = ~target_P
        if target_P.sum() == 0:
            # print("class", cls, "doesn't exist in target")
            continue
        else:
            intersection = pred_P[target_P].sum()  # TP
            if intersection == 0:
                # print("pred and target for class", cls, "have no intersection")
                continue
            else:
                FP = pred_P[target_N].sum()
                FN = pred_N[target_P].sum()
                union = intersection + FN + FP  # or pred_P.sum() + target_P.sum() - intersection
                cls_ious.append(float(intersection) / float(union))
    return cls_ious


def run(device, model, loader, criterion, is_train=False, optimizer=None):
    """
        Run forward pass for each sample in the dataloader. Run backward pass and optimize if training.
        Calculate and return mean_epoch_loss and mean_iou
        :param model (torch.nn.module object): miniUNet model object
        :param loader (torch.utils.data.DataLoader object): dataloader 
        :param criterion (torch.nn.module object): Pytorch criterion object
        :param is_train (bool): True if training
        :param optimizer (torch.optim.Optimizer object): Pytorch optimizer object
        :return mean_epoch_loss (float): mean loss across this epoch
        :return mean_iou (float): mean iou across this epoch
    """
    #model.train()
    # TODO: complete this function 
    # ===============================================================================
    total_loss, total_iou = 0, 0
    datalen = len(loader.dataset)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_train:
        model.train()
        for batch in loader:
            # Gets input and target data from batch, sends to device (in case using GPU)
            input = batch['input'].to(device)
            target = batch['target'].to(device)
            out = model(input)

            # Calculates metrics based on output
            loss = criterion(out, target)
            mIoU = iou(out, target, 4)
            total_loss += loss
            total_iou += sum(mIoU) 

            # Zeros gradients of optimizer, back calculates weights, and makes optimizer take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        model.eval()
        with torch.no_grad():
            for batch in loader:
            # Gets input and target data from batch, sends to device (in case using GPU)
                input = batch['input'].to(device)
                target = batch['target'].to(device)
                out = model(input)

                # Calculates metrics based on output
                loss = criterion(out, target)
                mIoU = iou(out, target, 4)
                total_loss += loss
                total_iou += sum(mIoU) 

    # Returning avg Loss/mIoU
    train_loss, train_iou = total_loss/datalen, total_iou/datalen
    return train_loss, train_iou
    # ===============================================================================

def convert_seg_split_into_color_image(img):
    color_palette = get_tableau_palette()
    colored_mask = np.zeros((*img.shape, 3))

    print(np.unique(img))

    for i, unique_val in enumerate(np.unique(img)):
        if unique_val == 0:
            obj_color = np.array([0, 0, 0])
        else:
            obj_color = np.array(color_palette[i-1]) * 255
        obj_pixel_indices = (img == unique_val)
        colored_mask[:, :, 0][obj_pixel_indices] = obj_color[0]
        colored_mask[:, :, 1][obj_pixel_indices] = obj_color[1]
        colored_mask[:, :, 2][obj_pixel_indices] = obj_color[2]
    return colored_mask.astype(np.uint8)


if __name__ == "__main__":
    # ==============Part 4 (a) Training Segmentation model ================
    # Complete all the TODO's in this file
    # - HINT: Most TODO's in this file are exactly the same as homework 2.
    batch_size = 8
    epoch, max_epochs = 1, 30

    seed(0)
    torch.manual_seed(0)

    # Check if GPU is being detected
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # TODO: Prepare train and test datasets
    # Load the "dataset" directory using RGBDataset class as a pytorch dataset
    # Split the above dataset into train and test dataset in 9:1 ratio using `torch.utils.data.random_split` method
    # ===============================================================================
    root_dir = './dataset/'

    seg_dataset = RGBDataset(root_dir)
    train_dataset, test_dataset = random_split(seg_dataset, [270, 30])
    #check_dataset(train_dataset)

    # ===============================================================================

    # TODO: Prepare train and test Dataloaders. Use appropriate batch size
    # ===============================================================================
    train_loader = DataLoader(train_dataset, batch_size, True) 
    test_loader = DataLoader(test_dataset, batch_size, False)
    #check_dataloader(train_loader)
    # ===============================================================================

    # TODO: Prepare model
    # ===============================================================================
    model = miniUNet(3, 4).to(device)

    # ===============================================================================

    # TODO: Define criterion, optimizer and learning rate scheduler
    # ===============================================================================
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #learning rate scheduler?
    # ===============================================================================

    # TODO: Train and test the model. 
    # Tips:
    # - Remember to save your model with best mIoU on objects using save_chkpt function
    # - Try to achieve Test mIoU >= 0.9 (Note: the value of 0.9 only makes sense if you have sufficiently large test set)
    # - Visualize the performance of a trained model using save_prediction method. Make sure that the predicted segmentation mask is almost correct.
    # ===============================================================================
    train_loss_list, train_miou_list, test_loss_list, test_miou_list = list(), list(), list(), list()
    best_miou = float('-inf')
    while epoch <= max_epochs:
        print('Epoch (', epoch, '/', max_epochs, ')')
        train_loss, train_miou = run(device, model, train_loader, criterion, True, optimizer)
        test_loss, test_miou = run(device, model, test_loader, criterion, False, optimizer)
        train_loss_list.append(train_loss)
        train_miou_list.append(train_miou)
        test_loss_list.append(test_loss)
        test_miou_list.append(test_miou)
        print('Train loss & mIoU: %0.2f %0.2f' % (train_loss, train_miou))
        print('Validation loss & mIoU: %0.2f %0.2f' % (test_loss, test_miou))
        print('---------------------------------')
        # Save the model with the best mIoU
        if test_miou > best_miou:
            best_miou = test_miou
            save_chkpt(model, epoch, test_miou, 'checkpoint.pth.tar')
        epoch += 1
    print(best_miou)
    # Load the best checkpoint, use save_prediction() on the validation set and test set
    model, epoch, best_miou = load_chkpt(model, 'checkpoint.pth.tar', device)
    #save_prediction(model, device, _loader, val_dir)
    save_prediction(model, test_loader, root_dir, device, batch_size)
    #save_learning_curve(train_loss_list, train_miou_list, test_loss_list, test_miou_list)
    # ===============================================================================