"""Import all necessary packages"""
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pathlib
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision  # modules and transforms for computer vision
from matplotlib import pyplot as plt
from tqdm import tqdm  # progress bar
import time
from torchsummary import summary
from Discriminator import Discriminator
from Generator import ResUNet as Generator
from Dataset import Dataset_cGAN

"""Import and load dataset"""
database = os.path.join(os.getcwd(), 'GLDataset2')
# Load the dataset
loader_train = torch.utils.data.DataLoader(
    dataset=Dataset_cGAN(database, 'train', transform=torchvision.transforms.Resize([512, 1024])),
    batch_size=1,
    shuffle=True)

loader_val = torch.utils.data.DataLoader(
    dataset=Dataset_cGAN(database, 'val', transform=torchvision.transforms.Resize([512, 1024])),
    batch_size=1,
    shuffle=True)

loader_test = torch.utils.data.DataLoader(
    dataset=Dataset_cGAN(database, 'test', transform=torchvision.transforms.Resize([512, 1024])))
# %%
"""Show the loaded image"""
nimages = 4
fig, axs = plt.subplots(ncols=nimages, nrows=2, figsize=(nimages * 4, 8))
i = 0
for images, labels in loader_test:
    for image, label in zip(images, labels):
        if i >= nimages:
            break
        axs[0, i].imshow(np.squeeze(image.numpy().transpose((1, 2, 0))), cmap='gray')
        axs[1, i].imshow(np.squeeze(label.numpy().transpose((1, 2, 0))), cmap='gray')
        axs[0, i].set_title(f'GL image {i + 1}', fontsize=10, pad=10)
        axs[1, i].set_title(f'NGL image {i + 1}', fontsize=10, pad=10)
        i += 1
plt.show()

"""Import generator, discriminator, optimizer and lose function"""
generator = Generator()
discriminator = Discriminator()

# Set the device to GPU if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# device = torch.device('cuda')
generator = generator.to(device)
discriminator = discriminator.to(device)


# Creat pixel level lose function
def L1_loss(output, target):
    loss = torch.mean(torch.abs(output - target))
    return loss


# learning rates
lr_gen = 1e-3
lr_dis = 1e-3

# define optimizers
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=lr_gen)
optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=lr_gen)

# define losses
pix_loss = L1_loss  # A loss penalizing difference in image space such as our L1 loss
dis_loss = torch.nn.BCEWithLogitsLoss()  # A loss adapted to binary classification like torch.nn.BCEWithLogitsLoss
lambda_pix = 1.  # weight for pix_loss
lambda_GAN = 1.  # weight for discriminator loss

# %%
"""Load the saved data"""
RESUME = True  # Controls whether training is resumed or not. False: initial training; True: continued training
start_epoch = -1
if RESUME:
    print('-----------------------------')
    path_checkpoint = os.path.join(os.getcwd(), 'checkpoints/2024_04_26_22_15_09_ckpt_120.pth')  # Check point path
    checkpoint = torch.load(path_checkpoint)  # load the check point
    generator.load_state_dict(checkpoint['generator'])  # Loading generator parameters
    discriminator.load_state_dict(checkpoint['discriminator'])  # Loading discriminator parameters
    optimizer_gen.load_state_dict(checkpoint['optimizer_gen'])  # Loading optimizer_gen parameters
    optimizer_dis.load_state_dict(checkpoint['optimizer_dis'])  # Loading optimizer_dis parameters
    losses_dis_train = checkpoint['losses_dis_train']  # Loading training losses - discriminator
    losses_gen_train = checkpoint['losses_gen_train']  # Loading training losses - generator
    losses_dis_val = checkpoint['losses_dis_val']  # Loading validation losses - discriminator
    losses_gen_val = checkpoint['losses_gen_val']  # Loading validation losses - generator
    start_epoch = checkpoint['epoch'] + 1  # Setting the start epoch
    print('Load epoch {} Succeed！'.format(start_epoch))
    print('-----------------------------')
else:
    start_epoch = 0
    losses_gen_train = []  # keep track of loss per epoch - generator
    losses_dis_train = []  # keep track of loss per epoch - discriminator
    losses_dis_val = []  # Loading validation losses - discriminator
    losses_gen_val = []  # Loading validation losses - generator
    print('No save model, will be trained from scratch!')
# %%
"""Model training & validation"""
TRAIN = False  # Controls whether train or not. False: no training; True: train
SAVE = False  # Controls whether save or not. False: not save; True: save
VALIDATION = False  # Controls whether use validation or not. False: not use; True: use
if TRAIN:
    num_epochs = 60  # number of epochs to training for
    # losses_gen_train = []  # keep track of training loss per epoch - generator
    # losses_dis_train = []  # keep track of training loss per epoch - discriminator
    # losses_gen_val = []  # keep track of validation loss per epoch - generator
    # losses_dis_val = []  # keep track of validation loss per epoch - discriminator
    for epoch in tqdm(range(start_epoch, start_epoch + num_epochs)):
        generator.train()  # switch to training mode
        discriminator.train()
        """Model training part"""
        loss_gen_train = 0
        loss_dis_train = 0
        loop_time_train = 0
        for images, labels in loader_train:
            optimizer_gen.zero_grad()  # zero optimizer gradients, default option accumulates!
            optimizer_dis.zero_grad()
            loop_time_train += 1
            ###############################
            # Inputs real T1 and real T2
            real_t1 = images.to(device)
            real_t2 = labels.to(device)
            # ---------------------
            #  Train Generator
            # ---------------------
            # Create labels
            valid = torch.as_tensor(np.ones((real_t2.size(0), 1, 1, 1)))
            fake = torch.as_tensor(np.zeros((real_t2.size(0), 1, 1, 1)))
            valid = valid.to(device)
            fake = fake.to(device)

            # Now synthesize a fake t2 image
            fake_t2 = generator(real_t1)
            # run it through the discriminator
            pred_fake = discriminator(fake_t2, real_t1)
            loss_GAN = dis_loss(pred_fake, valid)

            # Now the condition loss - supervised, this is condition part
            loss_pixel = pix_loss(fake_t2, real_t2)

            # Total loss
            loss_gen = lambda_GAN * loss_GAN + lambda_pix * loss_pixel

            # Compute the gradient and perform one optimization step
            loss_gen.backward()
            optimizer_gen.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_dis.zero_grad()

            # Real loss
            pred_real = discriminator(real_t2, real_t1)
            loss_real = dis_loss(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_t2.detach(), real_t1)
            loss_fake = dis_loss(pred_fake, fake)

            # Total loss
            loss_dis = 0.5 * (loss_real + loss_fake)

            # Compute the gradient and perform one optimization step
            loss_dis.backward()
            optimizer_dis.step()

            # keep track of training loss
            loss_gen_train += loss_gen.item()
            loss_dis_train += loss_dis.item()

        loss_gen_train /= (loader_train.batch_size * loop_time_train)  # normalize loss by batch size
        losses_gen_train.append(loss_gen_train)
        loss_dis_train /= (loader_train.batch_size * loop_time_train)  # optional, normalize loss by batch size
        losses_dis_train.append(loss_dis_train)
        # print('Current loss_dis_train:', loss_dis_train)
        # print('Current loss_gen_train:', loss_gen_train)

        """Validation part"""
        if VALIDATION:
            with torch.no_grad():
                loss_gen_val = 0
                loss_dis_val = 0
                loop_time_val = 0
                for images, labels in loader_val:
                    loop_time_val += 1
                    ###############################
                    # Inputs real T1 and real T2
                    real_t1 = images.to(device)
                    real_t2 = labels.to(device)
                    # ---------------------
                    #  Train Generator
                    # ---------------------
                    # Create labels
                    valid = torch.as_tensor(np.ones((real_t2.size(0), 1, 1, 1)))
                    fake = torch.as_tensor(np.zeros((real_t2.size(0), 1, 1, 1)))
                    valid = valid.to(device)
                    fake = fake.to(device)

                    # Now synthesize a fake t2 image
                    fake_t2 = generator(real_t1)
                    # run it through the discriminator
                    pred_fake = discriminator(fake_t2, real_t1)
                    loss_GAN = dis_loss(pred_fake, valid)

                    # Now the condition loss - supervised, this is condition part
                    loss_pixel = pix_loss(fake_t2, real_t2)

                    # Total loss
                    loss_gen = lambda_GAN * loss_GAN + lambda_pix * loss_pixel

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    # Real loss
                    pred_real = discriminator(real_t2, real_t1)
                    loss_real = dis_loss(pred_real, valid)

                    # Fake loss
                    pred_fake = discriminator(fake_t2.detach(), real_t1)
                    loss_fake = dis_loss(pred_fake, fake)

                    # Total loss
                    loss_dis = 0.5 * (loss_real + loss_fake)

                    # keep track of training loss
                    loss_gen_val += loss_gen.item()
                    loss_dis_val += loss_dis.item()

                loss_gen_val /= (loader_val.batch_size * loop_time_val)  # normalize loss by batch size
                losses_gen_val.append(loss_gen_val)
                loss_dis_val /= (loader_val.batch_size * loop_time_val)  # optional, normalize loss by batch size
                losses_dis_val.append(loss_dis_val)
                # print('Current loss_dis_val:', loss_dis_val)
                # print('Current loss_gen_val:', loss_gen_val)

        """Data save"""
        if SAVE:
            # Save the  parameter every 5 epochs
            if (epoch + 1) % 5 == 0:
                print('epoch:', epoch + 1)
                print('optimizer_gen learning rate:', optimizer_gen.state_dict()['param_groups'][0]['lr'])
                print('optimizer_dis learning rate:', optimizer_dis.state_dict()['param_groups'][0]['lr'])
                checkpoint = {
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'optimizer_gen': optimizer_gen.state_dict(),
                    'optimizer_dis': optimizer_dis.state_dict(),
                    'losses_gen_train': losses_gen_train,
                    'losses_dis_train': losses_dis_train,
                    'losses_gen_val': losses_gen_val,
                    'losses_dis_val': losses_dis_val,
                    'epoch': epoch
                }
                current_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                path_weights = os.path.join(os.getcwd(), 'checkpoints/%s_ckpt_%s.pth' % (current_time, str(epoch + 1)))
                torch.save(checkpoint, path_weights)
                print('Saved all parameters!\n')

# %%
"""Visualize the losses"""
# Train losses
fig, ax = plt.subplots(figsize=(6, 6))
lines1 = ax.plot(losses_gen_train, 'C0')
ax.set_ylabel('Gen Loss')
axr = ax.twinx()  # show Discriminator loss on the right axes
lines2 = axr.plot(losses_dis_train, 'C1')
axr.set_ylabel('Dis Loss')
ax.set_xlabel('Epoch')
ax.set_title('Training losses')  # Add title
ax.legend([lines1[0], lines2[0]], ['Gen', 'Dis'])
plt.show()
# Validation losses
fig_val, ax_val = plt.subplots(figsize=(6, 6))
lines1 = ax_val.plot(losses_gen_val, 'C0')
ax_val.set_ylabel('Gen Loss')
axr_val = ax_val.twinx()  # show Discriminator loss on the right axes
lines2 = axr_val.plot(losses_dis_val, 'C1')
axr_val.set_ylabel('Dis Loss')
ax_val.set_xlabel('Epoch')
ax_val.set_title('Validation losses')  # Add title
ax_val.legend([lines1[0], lines2[0]], ['Gen', 'Dis'])
plt.show()

# %%
"""Model test"""
images_test = np.empty((0, 1, 512, 1024))
labels_test = np.empty((0, 1, 512, 1024))
predictions_test = np.empty((0, 1, 512, 1024))
with torch.no_grad():
    generator.eval()
    for images, labels in tqdm(loader_test):
        images_test = np.append(images_test, images.numpy(), axis=0)
        outputs = generator(images.to(device))
        predictions = outputs.cpu()
        labels_test = np.append(labels_test, labels.numpy(), axis=0)
        predictions_test = np.append(predictions_test, predictions.numpy(), axis=0)
# %%
# Visualize the result
pred_std = np.std(predictions_test, axis=1)
nimages = 4
fig, axs = plt.subplots(ncols=nimages, nrows=3, figsize=(nimages * 4, 10))

for nIm in range(nimages):
    axs[0, nIm].imshow(np.squeeze(images_test[nIm, 0, :, :]), cmap='gray')
    axs[0, nIm].axis('off')  # Hide the axis
    axs[1, nIm].imshow(np.squeeze(predictions_test[nIm, 0, :, :]), cmap='gray')
    axs[1, nIm].axis('off')  # Hide the axis
    axs[2, nIm].imshow(np.squeeze(labels_test[nIm, :, :]), cmap='gray')
    axs[2, nIm].axis('off')  # Hide the axis
    # Set titles for each row
    axs[0, nIm].set_title(f'GL image {nIm + 1}', fontsize=10, pad=10)
    axs[1, nIm].set_title(f'Predicted NGL image {nIm + 1}', fontsize=10, pad=10)
    axs[2, nIm].set_title(f'Ground truth NGL image {nIm + 1}', fontsize=10, pad=10)
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)  # Adjust the space around the subplots
plt.show()
# %%
# Save and output the test result
OUTPUT = False  # Controls whether output or not. False: not output; True: output
if OUTPUT:
    path_result1 = os.path.join(os.getcwd(), 'Test_results/Predictions')
    path_result2 = os.path.join(os.getcwd(), 'Test_results/Labels')
    path_result3 = os.path.join(os.getcwd(), 'Test_results/Originals')

    if not os.path.exists(path_result1):
        os.makedirs(path_result1)
    if not os.path.exists(path_result2):
        os.makedirs(path_result2)
    if not os.path.exists(path_result3):
        os.makedirs(path_result3)

    for i in tqdm(range(len(images_test))):
        pre_img = np.squeeze(predictions_test[i, 0, :, :])
        pre_img = (pre_img * 255).astype(np.uint8)  # Convert to 8-bit unsigned integer
        cv2.imwrite(os.path.join(path_result1, 'Predict_%d.png' % i), pre_img)

        lab_img = np.squeeze(labels_test[i, :, :])
        lab_img = (lab_img * 255).astype(np.uint8)  # Convert to 8-bit unsigned integer
        cv2.imwrite(os.path.join(path_result2, 'Truth_%d.png' % i), lab_img)

        ori_img = np.squeeze(images_test[i, 0, :, :])
        ori_img = (ori_img * 255).astype(np.uint8)  # Convert to 8-bit unsigned integer
        cv2.imwrite(os.path.join(path_result3, 'Origin_%d.png' % i), ori_img)


print('finish')
