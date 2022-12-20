
#%%
# Import Required Modules
import os
import datetime
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import io
from IPython.display import clear_output
from tensorflow import keras
from keras import layers, losses, optimizers, callbacks
from keras.utils import plot_model, array_to_img
from keras.applications import MobileNetV2
from tensorflow_examples.models.pix2pix import pix2pix
from keras. callbacks import TensorBoard, EarlyStopping
from keras.metrics import Accuracy, IoU






#%%
# 1. Data Loading
# Create data path
root_path = os.path.join(os.getcwd(), "Dataset", "data-science-bowl-2018-2")
train_path = os.path.join(root_path, "train")
test_path = os.path.join(root_path, "test")

# An empty list for train images and masks
train_images = []
train_masks = []
test_images = []
test_masks = []

# Load train and test images using opencv
train_image_dir = os.path.join(train_path,'inputs')
for image_file in os.listdir(train_image_dir):
    img = cv2.imread(os.path.join(train_image_dir,image_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(128,128))
    train_images.append(img)
    
train_mask_dir = os.path.join(train_path,'masks')
for mask_file in os.listdir(train_mask_dir):
    mask = cv2.imread(os.path.join(train_mask_dir,mask_file),cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128,128))
    train_masks.append(mask)

test_image_dir = os.path.join(test_path,'inputs')
for image_file in os.listdir(test_image_dir):
    img = cv2.imread(os.path.join(test_image_dir,image_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(128,128))
    test_images.append(img)
    

test_mask_dir = os.path.join(test_path,'masks')
for mask_file in os.listdir(test_mask_dir):
    mask = cv2.imread(os.path.join(test_mask_dir,mask_file),cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128,128))
    test_masks.append(mask)
    
#%%
# Convert lists into numpy array
train_images_np = np.array(train_images)
train_masks_np = np.array(train_masks)
test_images_np = np.array(test_images)
test_masks_np = np.array(test_masks)

#%%
# Check some examples
plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(train_images_np[i])
    plt.axis('off')
    
plt.show()

plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(train_masks_np[i])
    plt.axis('off')
    
plt.show()

plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(test_images_np[i])
    plt.axis('off')
    
plt.show()

plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(test_masks_np[i])
    plt.axis('off')
    
plt.show()






#%%
# 2. Data preprocessing
# Expand mask dimension
train_masks_np_exp = np.expand_dims(train_masks_np,axis=-1)
test_masks_np_exp = np.expand_dims(test_masks_np,axis=-1)

# Check the mask output
print(np.unique(train_masks[0]))
print(np.unique(test_masks[0]))

# Convert the mask values into class labels
converted_train_masks = np.round(train_masks_np_exp/255).astype(np.int64)
converted_test_masks = np.round(test_masks_np_exp/255).astype(np.int64)

# Check the mask output
print(np.unique(converted_train_masks[0]))
print(np.unique(converted_test_masks[0]))

#%%
# Normalize image pixels value
converted_train_images = train_images_np / 255.0
sample = converted_train_images[0]
converted_test_images = test_images_np / 255.0
sample = converted_test_images[0]

#%%
# Convert numpy arrays into tensor 
x_train_tensor = tf.data.Dataset.from_tensor_slices(converted_train_images)
x_test_tensor = tf.data.Dataset.from_tensor_slices(converted_test_images)

y_train_tensor = tf.data.Dataset.from_tensor_slices(converted_train_masks)
y_test_tensor = tf.data.Dataset.from_tensor_slices(converted_test_masks)

#%%
# Combine images and masks using zip
train_dataset = tf.data.Dataset.zip((x_train_tensor,y_train_tensor))
test_dataset = tf.data.Dataset.zip((x_test_tensor,y_test_tensor))

#%%
# Create a subclass layer for data augmentation
class Augment(layers.Layer):
    def __init__(self,seed=42):
        super().__init__()
        self.augment_inputs = layers.RandomFlip(mode='horizontal',seed=seed)
        self.augment_labels = layers.RandomFlip(mode='horizontal',seed=seed)
        
    def call(self,inputs,labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs,labels
    
#%%
# Convert into prefetch 
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 250
TRAIN_SIZE = len(train_dataset)
STEPS_PER_EPOCH = TRAIN_SIZE//BATCH_SIZE

train_batches = (
    train_dataset
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

test_batches = test_dataset.batch(BATCH_SIZE)

#%%
# Visualize some examples
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image','True Mask','Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(array_to_img(display_list[i]))
        plt.axis('off')
        
    plt.show()
    
for images, masks in train_batches.take(2):
    sample_image,sample_mask = images[0],masks[0]
    display([sample_image,sample_mask])
    





#%%
# 3. Create image segmentation model
# Use a pretrained model as the feature extraction layers
base_model = MobileNetV2(input_shape=[128,128,3],include_top=False)

# List down some activation layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
    ]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Define the feature extraction model
down_stack = keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

# Define the upsampling path
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
    inputs = layers.Input(shape=[128,128,3])
    #A pply functional API to construct U-Net
    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    # Upsampling and establishing the skip connections(concatenation)
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = layers.Concatenate()
        x = concat([x,skip])
        
    # This is the last layer of the model (output layer)
    last = layers.Conv2DTranspose(
        filters=output_channels,
        kernel_size=3,
        strides=2,
        padding='same') #64x64 --> 128x128
    
    x = last(x)
    
    return keras.Model(inputs=inputs, outputs=x)

#%%
# Make of use of the function to construct the entire U-Net
OUTPUT_CLASSES = 2
model = unet_model(output_channels=OUTPUT_CLASSES)

# Compile the model
loss = losses.SparseCategoricalCrossentropy(from_logits=True)
iou = IoU(num_classes=2, target_class_ids=[1], sparse_y_pred=False)

model.compile(
    optimizer='adam',
    loss=loss,
    metrics=[iou,'accuracy'])

# Plot model
plot_model(model, show_shapes=True)

#%%
# Create functions to show predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],
                create_mask(pred_mask)])
            
    else:
        display([sample_image,
            sample_mask,
            create_mask(model.predict(sample_image[tf.newaxis,...]))])
        
#%%
# Test out the show_prediction function
show_predictions()

#%%
# Create a callback to help display results during model training
class DisplayCallback(callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))
        
# TensorBoard and EarlyStopping Callback 
logdir = os.path.join(os.getcwd(), 
    'logs', 
    datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tb = TensorBoard(log_dir=logdir)

es = EarlyStopping(monitor='l',
    patience=5)


#%%
# 4. Model training
# Hyperparameters for the model
EPOCHS = 20
VAL_SUBSPLITS = 2
VALIDATION_STEPS = len(test_dataset) // BATCH_SIZE // 2

history = model.fit(train_batches,
    validation_data=test_batches,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    callbacks=[es, tb, DisplayCallback()])






#%%
# 5. Deploy model
show_predictions(test_batches,3)

eval = model.evaluate(test_batches)
print(f" Loss : {eval[0]:.2f}\n IoU : {eval[1]:.2f}\n Accuracy : {eval[2]:.2f}\n")

model.save('saved_models.h5')


# %%
