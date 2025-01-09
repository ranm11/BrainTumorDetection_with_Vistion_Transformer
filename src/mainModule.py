from loadData import DatasetLoader
from VisionTransformer import ViT,img_to_patch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
import torchvision
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import os
DATASET_PATH = "brain_tumor_data"
CHECKPOINT_PATH = "saved_brain_tumor_models"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

def plot_metrics(metrics, title, ylabel):
    plt.plot(range(1, len(metrics) + 1), metrics, marker='o')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid()

def train_model(**kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "brainTumor_ViT"), 
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=100,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "Vit50.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = ViT.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(42) # To be reproducable
        model = ViT(**kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result ,trainer.logged_metrics

#load Dataset
loadInstance = DatasetLoader('C:\\Users\\ranmi\\dev\\GraphNeuralNet\\VisionTransformers\\brain_tumor_detection_transformers\\brain_tumor_dataset')
train_loader ,val_loader, test_loader, original_dataset,augmented_dataset  = loadInstance.loadDataset()
print("ZEVEL")
PLOT_ENABLE = 0
if PLOT_ENABLE:
    # yes-no plot
    plt.subplot(1, 2, 1)
    plt.title("Raw Image - no")
    no_img = original_dataset[47][0].permute(1, 2, 0)
    plt.imshow(no_img)   # 128 x 128 x 3
    
    plt.subplot(1, 2, 2)
    plt.title("Raw Image - yes")
    yes_img = original_dataset[100][0].permute(1, 2, 0)
    plt.imshow(yes_img)   # 128 x 128 x 3
    
    plt.show()
    plt.close()

    # image augmentation
    plt.subplot(1, 2, 1)
    plt.title("Original Image ")
    no_aug_img = original_dataset[47][0].permute(1, 2, 0)
    plt.imshow(no_aug_img)   # 128 x 128 x 3
    
    plt.subplot(1, 2, 2)
    plt.title("Augmented Image")
    yes_aug_img = augmented_dataset[47][0].permute(1, 2, 0)
    plt.imshow(yes_aug_img)   # 128 x 128 x 3
    
    plt.show()
    plt.close()

    #patch embedding
    # no_img = test_loader.dataset[2][0].permute(1, 2, 0)
    plt.imshow(no_img)   # 128 x 128 x 3
    plt.show()
    plt.close()

    NUM_IMAGES = len(train_loader.dataset)-1
    # dataset = brain_tumor_dataset.dataset
    #single image plot
    # all_imgs = torch.stack([dataset[idx][0] for idx in range(NUM_IMAGES)], dim=0)
    # all_labels = [dataset[idx][1] for idx in range(NUM_IMAGES)]
    # img_grid = torchvision.utils.make_grid(CIFAR_images, nrow=4, normalize=True, pad_value=0.9)
    # img_grid = img_grid.permute(1, 2, 0)

    #image patch plot
    # img_grid = torchvision.utils.make_grid(all_imgs, nrow=4, normalize=True, pad_value=0.9)
    # img_grid = img_grid.permute(1, 2, 0)
    all_imgs_plot = torch.stack([no_img,yes_img],dim=0)
    all_imgs_plot=all_imgs_plot.permute(0,3,1,2)
    img_patches = img_to_patch(all_imgs_plot, patch_size=8, flatten_channels=False)
    NOF_IMAGES_TO_PLOT = 2
    fig, ax = plt.subplots(1,NOF_IMAGES_TO_PLOT,  figsize=(14,3)) #all_imgs.shape[0]
    fig.suptitle("Images as input sequences of patches")
    for i in range(NOF_IMAGES_TO_PLOT):  #all_imgs.shape[0]
        img_grid = torchvision.utils.make_grid(img_patches[i], nrow=16, normalize=True, pad_value=0.9)
        img_grid = img_grid.permute(1, 2, 0)
        ax[i].imshow(img_grid)
        ax[i].axis('off')
    plt.show()
    plt.close()

else:
    
    
    model, results, metrics  = train_model(model_kwargs={
                                    'embed_dim': 256,
                                    'hidden_dim': 512,
                                    'num_heads': 8,
                                    'num_layers': 6,
                                    'patch_size': 8,
                                    'num_channels': 3,
                                    'num_patches': 256,
                                    'num_classes': 2,
                                    'dropout': 0.2
                                },
                                lr=3e-4)
print("ViT results", results)


all_tests_imgs = torch.stack([test_loader.dataset[idx][0] for idx in range(len(test_loader.dataset))], dim=0)
all_labels = torch.tensor([test_loader.dataset[idx][1] for idx in range(len(test_loader.dataset))])

model.eval()
with torch.no_grad():
    predictions = model(all_tests_imgs)

preds = torch.argmax(predictions)    

#watch image for misshit
from PIL import Image
 
# image_np = all_tests_imgs[1].permute(1, 2, 0).numpy()
image_np = (all_tests_imgs[11].permute(1, 2, 0) * 255).byte().numpy()

image = Image.fromarray(image_np)
image.save("tensor_image11.png")

#evaluate train and validation procedure
