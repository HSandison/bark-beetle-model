#!/usr/bin/env python
# coding: utf-8

# This code trains and evaluates a model using the 'Combined Images for Model' folder in SuperAnnotate.

# In[ ]:





# In[ ]:





# # --------------------------------------------------------------------------------
# # Bark Beetle Detection using Detectron2
# # Developer: Helen Sandison
# # Date: 15th October 2024
# # Filename: bark_beetle_detection_v1.5
# # Version: v1.5
# # Kaggle Notebook
# # --------------------------------------------------------------------------------
# # Description:
# # This notebook implements a Mask R-CNN model using Detectron2 to detect bark beetles from trap contents transferred to lab petri dishes. It performs dataset preparation, model training, and inference.
# # Dataset structure is organised within Google Drive for easy access.
# # --------------------------------------------------------------------------------
# # Changelog:
# # - v1.1: Remove redundant code
# # Updated path definitions and usage
# # Added in try-catch errors and logging
# # Saved checkpoints every 500 iterations to Google Drive - and model checks and picks up from where it left off by checking Google Drive
# # v1.2: Move notebook to Kaggle
# # Still trying to save checkpoints to Google Drive 
# # Kaggle GPU availability = 30 hours per week
# # v1.5 larger dataset bark-beetle-dataset-v5
# # --------------------------------------------------------------------------------
# # Dataset Info
# # Folders combined in SuperAnnotate:
# # 14_8_to_16_08_F: all files except ER60_4_6_8.jpg [5 items]
# # 21_08_to_02_09 F: all files [18 items]
# # Composite images: all files [21 items]
# # square dish - camera help by hand: all files [7 items]
# # 51 image files in total
# # Train=36; val=6; test=9
# # Downloaded in COCO format, for instance segmentation
# --------------------------------------------------------------------------------

# In[1]:


# Install required libraries

# Install PyTorch for GPU
get_ipython().system('pip install torch torchvision torchaudio')

# Install Detectron2
get_ipython().system('pip install -U torch torchvision')
get_ipython().system("pip install 'git+https://github.com/facebookresearch/detectron2.git'")

# Install OpenCV and pycocotools for handling images and COCO format
get_ipython().system('pip install opencv-python-headless')
get_ipython().system('pip install pycocotools')


# In[2]:


# Import Libraries

# Import libraries
import os
import json
import random
import numpy as np
import cv2
import logging
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
setup_logger()

# Setup a general logger for error handling
logging.basicConfig(filename="error_log.log", level=logging.ERROR,
                    format="%(asctime)s - %(levelname)s - %(message)s")


# In[4]:


# Define important directories

# Define dataset directory
dataset_dir = Path("/kaggle/input/bark-beetle-dataset-v5/bark-beetle-dataset-v5")  # Update this path if needed

# Define directories for images and annotations
train_images_dir = dataset_dir / 'train' / 'images'
train_annotations_file = dataset_dir / 'train' / 'annotations' / "coco_annotations_train.json"

val_images_dir = dataset_dir / 'val' / 'images'
val_annotations_file = dataset_dir / 'val' / 'annotations' / "coco_annotations_val.json"

test_images_dir = dataset_dir / 'test' / 'images'
test_annotations_file = dataset_dir / 'test' / 'annotations' / "coco_annotations_test.json"

# Verify that the annotation files exist
for file in [train_annotations_file, val_annotations_file, test_annotations_file]:
    if not file.exists():
        logging.error(f"Annotation file not found: {file}")


# In[5]:


import os

# Recursively print all files and directories in the dataset directory
for root, dirs, files in os.walk(dataset_dir):
    print(f"\nRoot: {root}")
    print(f"Directories: {dirs}")
    print(f"Files: {files}")



# In[6]:


from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Function to unregister existing datasets
def unregister_datasets():
    # List of dataset names to unregister
    dataset_names = ["bark_beetle_train", "bark_beetle_val", "bark_beetle_test"]
    
    # Unregister datasets
    for name in dataset_names:
        # Check and remove dataset from DatasetCatalog
        if name in DatasetCatalog.list():  # Use list() to get registered dataset names
            DatasetCatalog.remove(name)
        # Check and remove dataset from MetadataCatalog
        if name in MetadataCatalog.list():  # Use list() to get registered dataset names
            MetadataCatalog.remove(name)

# Unregister the datasets
unregister_datasets()

# Register the datasets using the available annotation files
register_coco_instances("bark_beetle_train", {}, str(train_annotations_file), str(train_images_dir))
register_coco_instances("bark_beetle_val", {}, str(val_annotations_file), str(val_images_dir))
register_coco_instances("bark_beetle_test", {}, str(test_annotations_file), str(test_images_dir))

# Verify registration
train_metadata = MetadataCatalog.get("bark_beetle_train")
val_metadata = MetadataCatalog.get("bark_beetle_val")

print("Registered training dataset with", len(DatasetCatalog.get("bark_beetle_train")), "images.")
print("Registered validation dataset with", len(DatasetCatalog.get("bark_beetle_val")), "images.")


# In[ ]:


# visualise dataset

import random
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Visualize training images and annotations
def visualize_dataset(dataset_name, num_images=5):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    for d in random.sample(dataset_dicts, num_images):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata)
        out = visualizer.draw_dataset_dict(d)
        plt.figure(figsize=(10, 10))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.axis('off')
        plt.show()

# Visualize training and validation datasets
visualize_dataset("bark_beetle_train")
visualize_dataset("bark_beetle_val")


# In[9]:


# CLEAR THE OLD OUTPUT AND CHECKPOINTS DIRECTORY AND CREATE NEW ONES

import shutil
import os

# Clear the output directory if it exists
output_dir = './output'  # Adjust this if your output path is different
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # This will delete the entire directory and its contents
    print("Old output directory cleared.")

# Clear the checkpoints directory if it exists
checkpoints_dir = './checkpoints'  # Adjust this if your output path is different
if os.path.exists(checkpoints_dir):
    shutil.rmtree(checkpoints_dir)  # This will delete the entire directory and its contents
    print("Old checkpoints directory cleared.")

os.makedirs(output_dir, exist_ok=True)  # This creates the directory if it doesn't exist
os.makedirs(checkpoints_dir, exist_ok=True)  # This creates the directory if it doesn't exist


# In[12]:


# Configure the model for training
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Initialize configuration
cfg = get_cfg()

# Load the Mask R-CNN model configuration
# 1st model: mask_rcnn_R_50_FPN_3x.yaml
# 2nd model: mask_rcnn_R_101_FPN_3x.yaml
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

# Set the dataset names
cfg.DATASETS.TRAIN = ("bark_beetle_train",) # Training dataset
cfg.DATASETS.TEST = ("bark_beetle_val",)  # Validation dataset

# Data loader settings
cfg.DATALOADER.NUM_WORKERS = 2  # Number of workers for data loading

# Solver settings
cfg.SOLVER.IMS_PER_BATCH = 2  # Batch size
cfg.SOLVER.BASE_LR = 0.00025  # Learning rate
cfg.SOLVER.MAX_ITER = 3000  # Number of iterations

# Checkpoint settings
cfg.SOLVER.CHECKPOINT_PERIOD = 500  # Save checkpoint every 50 iterations

# Output directory
cfg.MODEL.OUTPUT_DIR = "./output"  # Directory to save model outputs

# Set the number of classes in the model (including the background)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(train_metadata.get("thing_classes", []))

# Create output directory
os.makedirs(cfg.MODEL.OUTPUT_DIR, exist_ok=True)

# Checkpoint directory for Kaggle
kaggle_checkpoint_dir = Path("./checkpoints")  # Local directory for saving checkpoints

# Create the local directory if it doesn't exist
os.makedirs(kaggle_checkpoint_dir, exist_ok=True)


# In[ ]:


import shutil
import os
from detectron2.engine import DefaultTrainer
from pathlib import Path

# Define output and checkpoints directories
output_dir = "/kaggle/working/output"
checkpoint_dir = "/kaggle/working/checkpoints"

# Ensure output and checkpoints directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Print current working directory for debugging
print("Current working directory:", os.getcwd())

# Create a trainer object
trainer = DefaultTrainer(cfg)  # Initialise trainer
trainer.resume_or_load(resume=False)  # Do not resume from previous checkpoint

# Start training
trainer.train()

# Immediately copy the latest checkpoint to the output directory after training
def copy_latest_checkpoint():
    checkpoints = sorted(Path(checkpoint_dir).glob("*.pth"), key=os.path.getmtime)
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        shutil.copy(latest_checkpoint, output_dir)
        print(f"Latest checkpoint copied to output directory: {latest_checkpoint.name}")
    else:
        print("No checkpoints found to copy.")

# Call the function to copy the latest checkpoint
copy_latest_checkpoint()

# List contents of /kaggle/working/ to verify if directories and files exist
print("Contents of /kaggle/working/:")
print(os.listdir("/kaggle/working/"))



# In[11]:


import shutil

# Create a zip file of the output directory
shutil.make_archive('/kaggle/working/model_output', 'zip', './output')


# In[ ]:


import os

# Create a directory for the model
os.makedirs('/kaggle/working/model_directory', exist_ok=True)

# Move the model_final.pth file to the new directory
shutil.copy('./output/model_final.pth', '/kaggle/working/model_directory/model_final.pth')

# Initialize the Kaggle dataset
get_ipython().system('kaggle datasets init -p /kaggle/working/model_directory')

# Create the Kaggle dataset
get_ipython().system('kaggle datasets create -p /kaggle/working/model_directory --dir-mode zip')


# In[12]:


# Check for GPU availability
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("GPU is available!")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available.")


# In[13]:


import multiprocessing
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import inference_on_dataset

# Set the start method for multiprocessing to avoid potential deadlocks
multiprocessing.set_start_method('spawn', force=True)

# Create COCO Evaluator for validation set
evaluator = COCOEvaluator(
    dataset_name="bark_beetle_val",
    output_dir="./output/",
    use_fast_impl=False  # Set to True to use a faster implementation if supported
)

# Build the validation data loader
val_loader = build_detection_test_loader(cfg, "bark_beetle_val")

# Evaluate the model on the validation set
results = inference_on_dataset(trainer.model, val_loader, evaluator)
print("Evaluation results on validation set:", results)




# In[14]:


# Import necessary libraries
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer

# Initialize configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# Specify the correct number of classes
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # Change this to your actual number of classes

# Load the weights from the trained model
cfg.MODEL.WEIGHTS = './output/model_final.pth'  # Adjust the path if needed
# If using Google Drive, uncomment the next line and comment the line above
# cfg.MODEL.WEIGHTS = '/content/drive/MyDrive/BarkBeetleDetectionModel/model/model_final.pth'

# Set the threshold for predictions
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the threshold for this model

# Force the code to run on CPU
cfg.MODEL.DEVICE = "cuda"  # cpu - or change this to "cuda" if you have GPU support

# Create predictor
predictor = DefaultPredictor(cfg)

# Create checkpointer and load weights (without weights_only)
checkpointer = DetectionCheckpointer(predictor.model)
checkpointer.load(cfg.MODEL.WEIGHTS)  # Removed weights_only parameter


# In[15]:


import cv2 
import os 
import matplotlib.pyplot as plt 
from detectron2.utils.visualizer import Visualizer 
from detectron2.data import MetadataCatalog

# Correct image folder path for Kaggle
image_folder_path = "/kaggle/input/bark-beetle-dataset-v5/bark-beetle-dataset-v5/test/images/" 
predictions_folder_path = "./predictions/"  # Local predictions folder within the working directory

# Create predictions folder if it doesn't exist
os.makedirs(predictions_folder_path, exist_ok=True)

# List images in the folder to ensure it contains the expected files
print("Listing images in the folder:")
image_files = os.listdir(image_folder_path)  # List images in the folder
print(image_files)

# Loop through each image in the directory
for image_file in image_files:
    sample_image_path = os.path.join(image_folder_path, image_file)  # Get the full image path
    image = cv2.imread(sample_image_path)

    # Check if the image was loaded correctly
    if image is None: 
        print(f"Error loading the image: {sample_image_path}")
        continue  # Skip to the next image if there is an error

    print(f"Loaded image: {sample_image_path}")  # Print loaded image path

    # Make predictions
    outputs = predictor(image)

    # Check predictions
    print("Predictions:", outputs)  # Inspect the outputs
    if outputs["instances"].has("pred_classes"):
        print("Number of instances predicted:", len(outputs["instances"]))
    else:
        print("No instances predicted.")

    # Visualize the predictions
    metadata = MetadataCatalog.get("bark_beetle_val")
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Create a new filename with '_prediction' suffix
    new_filename = os.path.splitext(image_file)[0] + "_prediction.jpg"  # Append '_prediction' to the name
    saved_image_path = os.path.join(predictions_folder_path, new_filename)  # Create new file path

    # Save the visualized image with predictions
    cv2.imwrite(saved_image_path, v.get_image()[:, :, ::-1])  # Save the image in BGR format
    print(f"Saved image with predictions to: {saved_image_path}")

    # Create directories for each predicted class
    instances = outputs["instances"].to("cpu")
    classes = instances.pred_classes.numpy()  # Get predicted classes
    boxes = instances.pred_boxes.tensor.numpy()  # Get bounding boxes

    # Get the unique classes
    unique_classes = set(classes)
    class_folders = {}

    # Create a folder for each class
    for cls in unique_classes:
        class_name = metadata.thing_classes[cls]  # Get class name
        class_folder_path = os.path.join(predictions_folder_path, class_name)
        os.makedirs(class_folder_path, exist_ok=True)
        class_folders[class_name] = class_folder_path  # Store the class folder path

    # Extract and save each prediction
    for i in range(len(classes)):
        class_name = metadata.thing_classes[classes[i]]
        box = boxes[i].astype(int)  # Convert to integer for indexing

        # Extract the bounding box
        x1, y1, x2, y2 = box
        extracted_image = image[y1:y2, x1:x2]

        # Create a unique filename for each extracted object
        extracted_filename = os.path.join(class_folders[class_name], f"{image_file}_class_{class_name}_{i}.jpg")
        cv2.imwrite(extracted_filename, extracted_image)  # Save the extracted image
        print(f"Saved extracted image to: {extracted_filename}")

    # Show the image with predictions using matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(v.get_image()[:, :, ::-1])  # Convert BGR to RGB
    plt.axis('off')  # Hide axes
    plt.show()


# In[1]:


# Display all predictions of a class in a grid format

import os
import cv2
import matplotlib.pyplot as plt
import math

# Set the path to the folder containing images
ips_folder_path = "./predictions/Ips/"

# Set the path to the 'grids' folder (same level as 'Ips')
grids_folder_path = "./predictions/grids/"

# Check if the folder exists, create it if not
if not os.path.exists(grids_folder_path):
    os.makedirs(grids_folder_path)
    print(f"Created folder: {grids_folder_path}")

# Check if the 'Ips' folder exists
if not os.path.exists(ips_folder_path):
    print("The specified 'Ips' folder does not exist.")
else:
    # List all image files in the folder
    image_files = [f for f in os.listdir(ips_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Check if there are images to display
    if not image_files:
        print("No images found in the specified 'Ips' folder.")
    else:
        # Calculate grid size
        num_images = len(image_files)
        grid_size = math.ceil(math.sqrt(num_images))  # Determine grid size (e.g., 4x4 for 16 images)

        # Create subplots
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()  # Flatten to easily access axes in a loop

        for i, image_file in enumerate(image_files):
            # Construct the full path to the image
            image_path = os.path.join(ips_folder_path, image_file)

            # Read the image using OpenCV
            image = cv2.imread(image_path)

            # Check if the image was loaded successfully
            if image is None:
                print(f"Error loading image: {image_path}")
                continue

            # Convert the image from BGR to RGB (OpenCV loads images in BGR format)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display the image
            axes[i].imshow(image_rgb)
            axes[i].axis('off')  # Hide the axis
            axes[i].set_title(image_file)  # Show the image file name

        # Hide any remaining empty subplots
        for j in range(i + 1, grid_size * grid_size):
            axes[j].axis('off')

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save the grid to the 'grids' folder with a given name
        output_file_path = os.path.join(grids_folder_path, "predictions_grid.png")
        plt.savefig(output_file_path)

        # Show the grid
        plt.show()

        print(f"Grid of predictions saved as: {output_file_path}")


# In[ ]:


get_ipython().system('kaggle datasets init -p /kaggle/working/')


# In[ ]:


import shutil

# Specify the path to the predictions folder
predictions_folder_path = "./predictions/"

# Specify the name for the zip file
zip_file_name = "predictions.zip"

# Create a zip file of the predictions folder
shutil.make_archive(zip_file_name.replace('.zip', ''), 'zip', predictions_folder_path)

print(f"Zipped predictions folder into {zip_file_name}")


# # Import necessary libraries
# from detectron2.config import get_cfg
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.checkpoint import DetectionCheckpointer  # Import DetectionCheckpointer
# 
# # Initialize configuration
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# 
# # Specify the correct number of classes
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # Change this to your actual number of classes
# 
# # # Load the weights from the trained model
# # cfg.MODEL.WEIGHTS = './output/model_final.pth'  # Adjust the path if needed
# # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the threshold for this model
# 
# # Load the weights from Google Drive
# cfg.MODEL.WEIGHTS = '/content/drive/MyDrive/BarkBeetleDetectionModel/model/model_final.pth'
# 
# # Force the code to run on CPU
# cfg.MODEL.DEVICE = "cpu"
# 
# # Create predictor
# predictor = DefaultPredictor(cfg)
# 
# # Create checkpointer and load weights (without weights_only)
# checkpointer = DetectionCheckpointer(predictor.model)
# checkpointer.load(cfg.MODEL.WEIGHTS)  # Removed weights_only parameter
# 
# 

# import cv2
# import random
# import os
# import matplotlib.pyplot as plt
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog
# 
# # Correct image folder path
# image_folder_path = "/content/drive/MyDrive/BarkBeetleDetectionModel/data/images/test/"
# predictions_folder_path = "/content/drive/MyDrive/BarkBeetleDetectionModel/predictions/"
# 
# # Create predictions folder if it doesn't exist
# os.makedirs(predictions_folder_path, exist_ok=True)
# 
# # List images in the folder to ensure it contains the expected files
# print("Listing images in the folder:")
# image_files = os.listdir(image_folder_path)  # List images in the folder
# print(image_files)
# 
# # Load a random image from the directory
# sample_image_path = os.path.join(image_folder_path, random.choice(image_files))  # Choose a random image
# #sample_image_path = os.path.join(image_folder_path, ('WE203_26_2.jpg'))
# image = cv2.imread(sample_image_path)
# 
# # Check if the image was loaded correctly
# if image is None:
#     print("Error loading the image.")
# else:
#     print(f"Loaded image: {sample_image_path}")  # Print loaded image path
# 
#     # Make predictions
#     outputs = predictor(image)
# 
#     # Check predictions
#     print("Predictions:", outputs)  # Inspect the outputs
#     if outputs["instances"].has("pred_classes"):
#         print("Number of instances predicted:", len(outputs["instances"]))
#     else:
#         print("No instances predicted.")
# 
#     # Visualize the predictions
#     metadata = MetadataCatalog.get("bark_beetle_val")
#     v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
#     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# 
#     # Create a new filename with '_prediction' suffix
#     base_filename = os.path.basename(sample_image_path)  # Get original filename
#     new_filename = os.path.splitext(base_filename)[0] + "_prediction.jpg"  # Append '_prediction' to the name
#     saved_image_path = os.path.join(predictions_folder_path, new_filename)  # Create new file path
# 
#     # Save the visualized image with predictions
#     cv2.imwrite(saved_image_path, v.get_image()[:, :, ::-1])  # Save the image in BGR format
# 
#     print(f"Saved image with predictions to: {saved_image_path}")
# 
#     # Create directories for each predicted class
#     instances = outputs["instances"].to("cpu")
#     classes = instances.pred_classes.numpy()  # Get predicted classes
#     boxes = instances.pred_boxes.tensor.numpy()  # Get bounding boxes
# 
#     # Get the unique classes
#     unique_classes = set(classes)
#     class_folders = {}
# 
#     # Create a folder for each class
#     for cls in unique_classes:
#         class_name = metadata.thing_classes[cls]  # Get class name
#         class_folder_path = os.path.join(predictions_folder_path, class_name)
#         os.makedirs(class_folder_path, exist_ok=True)
#         class_folders[class_name] = class_folder_path  # Store the class folder path
# 
#     # Extract and save each prediction
#     for i in range(len(classes)):
#         class_name = metadata.thing_classes[classes[i]]
#         box = boxes[i].astype(int)  # Convert to integer for indexing
# 
#         # Extract the bounding box
#         x1, y1, x2, y2 = box
#         extracted_image = image[y1:y2, x1:x2]
# 
#         # Create a unique filename for each extracted object
#         extracted_filename = os.path.join(class_folders[class_name], f"{base_filename}_class_{class_name}_{i}.jpg")
#         cv2.imwrite(extracted_filename, extracted_image)  # Save the extracted image
# 
#         print(f"Saved extracted image to: {extracted_filename}")
# 
#     # Show the image with predictions using matplotlib
#     plt.figure(figsize=(12, 8))
#     plt.imshow(v.get_image()[:, :, ::-1])  # Convert BGR to RGB
#     plt.axis('off')  # Hide axes
#     plt.show()
# 

# 

# In[ ]:




