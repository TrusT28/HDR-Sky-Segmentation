# Report pipeline

# Pipeline Report

# Contents

- Technologies, Libraries
    - Python 3.7 and 3.9
    - Virtual environment
        - Conda
    - CUDA
    - Used Libraries
- Source of dataset
    - Where dataset was created
        - label-studio, polygons
    - Parsing dataset
        - COCO
- Processing dataset
    - Preprocessing images (normal, HDR)
        - 1./255 and log transofrmations
        - .tif for HDR, .npy
    - Creating patches
    - Splitting into training/validation
        - train_test_split
        - 0.75/0.25
    - Generators
        - ImageDataGenerator and its abilities
        - Custom for HDR
    - Augmentations ( on normal pictures, plans for HDR )
        - Generator + augmentation
        - List augmentations made
    - Future plans
        - Add augmentation for custom HDR generator
            - Gamma
        - Better preprocesing of HDR (sun clipping)
        - Test different architectures, apply layers accordingly
- Code organization
    - Booleans
    - Shell script
- Model
    - Architecture: resnet34 and why
    - Metrics used: loss, iou
    - segmentation_models
    - callbacks
- Prediction
    - Splitting image
    - predicting
    - unpatching result, saving
    - combining in photoshop (change in future)

---

# Technologies and Libraries

In this work primary programming language is ***Python***. Most of the work was done on ***Python** **3.7***, but now the code is moved to ***Python 3.9***, which will be used for the final version of the project.

The package and environment manager of choice is ***Conda***. Two primary environments were created for this work—one for working in ***Python 3.7*** and another for the latest transformation to ***Python 3.9***. Every python library is installed using ***PIP*** in the corresponding ***Conda*** environment. Many libraries are created for modern Deep Learning, and most of them are very sensitive to the versions used of the programming language, ***CUDA*** driver, and other libraries it works with. Therefore, using virtual environments to support all different combinations is essential.

The Main Deep Learning libraries used are ***Tensorflow*** and ***Keras***. ML model, and most of the pipeline is built on them. Originally, ***Tensorflow 2.4.1*** and independent(not within Tensorflow) ***Keras 2.3.1*** were used with ***Python 3.7***. The main reason for that choice was the requirements of the ***Segmentation_modles*** library. Currently, the code was moved to use ***Tensorflow 2.9.1***, the corresponding Keras version within Tensorflow. The primary mathematics and data array manipulation tool used is ***Numpy***. Model’s Backbone is taken from [***Segmentation_models](https://github.com/qubvel/segmentation_models)*** library.

***CUDA*** toolkit and ***CUDDN*** are installed manually based on the Tensorflow version used. ***CUDA 10.1*** was used for ***Tensorflow 2.4.1***.

List of used libraries:

- Tensorflow (2.4.1 and 2.9.1)
- Keras (2.3.1 and corresponding tensorflow)
- Numpy
- cv2
- matplotlib - plotting images, saving images
- Segmentation_models - backbone of the model
- pytinyexr - reading .exr files
- tifffile - reading and saving .tif files
- PIL - reading and saving every other image files
- pycocotools - parsing COCO .json file
- sklearn, train_test_split - automatical random split into training and validation sets
- ImageDataGenerator - Keras Image Generator

---

# Dataset

## Source of dataset

For this project, a custom dataset is used. It combines sky images from [the Polyhaven](https://polyhaven.com/hdris/skies) project and the university's internal datasets. Images use ***.exr*** format for HDR and ***.jpg*** for LDR.

At this point, the dataset consists of 46 images from the university’s dataset and 29 images from Polyhaven, resulting in a total of 75 images. All images are in 8K (8192x4096).

Extra pictures will expand Dataset.

## Labeling

All images were mainly labeled using the ***[Label-studio](https://labelstud.io/)*** tool. It accepts only LDR images. Images were annotated using polygons.

Since the task is binary classification, a class for the sky was created (value = 1). Everything that is not labeled as sky is considered non-sky (value = 0). Another non-sky class was created (value = 2) to label objects which appear in the middle of the sky. Later in the code, this extra non-sky class with value=2 is changed to just 0.

## Parsing

Labeled images are saved in [COCO format](https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html) as ***.json*** file with polygons for images and folder of original images. For python, ***pycocotools*** parses COCO ***.json*** file and creates masks out of polygons for corresponding image. Such mask is an array of needed size (in this case 8192x4096), which has values of 1 (sky) or 0 (non-sky). 

In the code, parsing is done in functions “`createInitialHDRDataset`” for HDR images and “`getInitialDataset`” for LDR images.

Results of parsing are saved into corresponding pairs of original image and its mask. It is always assumed to be true for the later operations on the dataset.

## Processing Dataset

### Pre-processing

For both LDR and HDR images, pre-processing is very important for correct training. For LDR images it is enough to divide pixel values by 255 to put values in the comfortable for the network range between 0 and 1 (normalization). It results in float32 data type. HDR images require more complex transformations. By suggestions, the following log transformation worked the best in other deep learning experiments:

```python
def preProcessHDRdata(image):
    transform_cfg = dnnlib.EasyDict(
        # Multiplier value applied before the log transform
        input_mul = 1, # e.g. 2**8 would shift it 8 EV steps up
        # The epsilon constant for log-mapping input HDR images.
        log_epsilon = 1e-3,
        # separate the value space around 1.0 into two separate mapping functions
        # - x < 1: log(x) => expansion of the value space
        # - x >= 1: pow(x, 1./log_pow) => adjustable compression of the value space
        log_split_around1 = False,
        log_pow = 7.5,
        # Shift the value up/down (after the log transform) - can be used to ensure zero-mean
        output_bias = 2.5,
        output_scale = 1/2.2/2,
        )
    new_image = log_transform(image, transform_cfg)
    return new_image
    
# Applies the log-transformation that converts linear HDR images to a form suitable for a
# neural network.
def log_transform(x, transform_cfg):
    if transform_cfg.input_mul != 1.0:
        x = x * transform_cfg.input_mul
    x = x + transform_cfg.log_epsilon

    log_x = np.log(x)
    pow_x = np.power(x, 1./transform_cfg.log_pow) - 1.0

    if transform_cfg.log_split_around1:
        x = np.where(x < 1.0, log_x, pow_x)
    else:
        x = log_x
    return (x + transform_cfg.output_bias) * transform_cfg.output_scale

# Inverse of log_transform(x)
def invert_log_transform(y, transform_cfg):
    y = y / transform_cfg.output_scale - transform_cfg.output_bias

    exp_y = np.exp(y)
    pow_y = np.power(y + 1.0, transform_cfg.log_pow)

    if transform_cfg.log_split_around1:
        y = np.where(y < 0.0/transform_cfg.input_mul, exp_y, pow_y)
    else:
        y = exp_y
    y = y - transform_cfg.log_epsilon
    
    if transform_cfg.input_mul != 1.0:
        y = y / transform_cfg.input_mul
    return y
```

Alpha channel of HDR images is ignored (so far it is 1.0 in all images, so it does not matter).

### Patching

After pre-processing, all images and corresponding masks are patched (divided into smaller images, patches). Experiments were done with 512x512 patches for LDR images. For HDR images, it is 256x256 patches with 100 pixels stride and valid padding (meaning we don’t add padding to images). Patches are created using ***the TensorFlow*** function ***extract_patches,*** which expects a 4-dimensional array (we can create patches for several pictures simultaneously).

```python
patches_image = tf.image.extract_patches(images=image,
                            sizes=[1, SIZE_X, SIZE_Y, 1],
                            strides=[1, STEP, STEP, 1],
                            rates=[1, 1, 1, 1],
                            padding='VALID')
```

When predicting image using trained model, it is assumed that image is also patched into correct sized patches (256x256 for HDR images).

Later, we can “unpatchify” the image back into its full size by combining patches in the correct order. This can be done using library ***patchify***, which has the method ***unpatchify.*** This process can be tricky and works when stride was equal to the size of the patch (so patches do not overlap). 

```python
(w, l, d, x1, y1, c) = original_patches.shape
predicted = np.reshape(predicted, (w, l, x1, y1))
predicted = unpatchify(predicted, (x,y))
final_prediction = predicted.reshape(x,y)
```

TODO: Writting a custom method which would unpatchify overlapping patches and know where patches overlap can improve the final prediction results, since we can combine results from 2 or more predictions where 2 different patches overlapped.

### Splitting into training and validation sets

The patches dataset is randomly split into training and validation sets using ***train_test_split*** library. The split ratio is 75%/25%, respectively.

Especially for HDR images, such datasets can be too huge to fit into the RAM or GPU memory for training. Therefore, splitted into training and validation sets patches are saved into the folder as files, which will be later loaded lazily when needed.

Folder structure:

- X_train
    - train
        - 1.tif
        - 2.tif
        - …
- Y_train
    - train
        - 1.png
        - 2.png
        - …
- X_val
    - val
        - 3.tif
        - 4.tif
        - …
- Y_val
    - val
        - 3.png
        - 4.png
        - …

X stands for original images, Y stands for masks.

Mostly libraries assume that inside every folder will be folders with classes, inside of which will be files. Since we have binary semantic segmentation task and we already have all needed information inside the mask file, we just put one folder with random name (train for training set and val for validation set).

### Saving processed images for later training

Since HDR images in .exr format are float32 images which store a lot of information, it is important to save processed images in **lossless** format. Even though, they can be stored in ***.npy*** format, currently ***.tif*** format is used as the most popular.

To save splitted and patched HDR images, we use ***.tif*** format and library ***tifffile***

```python
from tifffile import imsave as tifImsave
tifImsave('X_train/train/1.tif',single_patch_of_hdr)
```

Because of memory restrictions, creation of HDR dataset is different than of LDR dataset. For HDR dataset one ***.exr*** file is loaded at a time, patched, splitted, saved and deleted from the memory. For LDR we can allow to load all images into memory at once and save them later.

LDR images can be save using ***PIL*** library in the similar way as ***.jpg***

The problem with multidimensional float32 .tif images is that most of the image libraries, even widely used library ***PIL***, do not support them. Opening such files is done via ***tifffile*** library. It results in the problem that available generators, which are supposed to read images from directory and feed them to the network, cannot be used. So for HDR images, the custom generator was written.

## Generators and Augmentations

### LDR

For LDR images, I used ***Keras ImageDataGenerator,*** which can read many LDR files (it uses ***PIL*** for that). 

```python
image_data_generator = ImageDataGenerator(**img_data_gen_args)
image_generator = image_data_generator.flow_from_directory(DATA_DIR+'x_train/',
                                                           seed=seed,
                                                           batch_size=batch_size,
                                                           target_size=(512, 512),
                                                           class_mode=None
                                                          )  #Very important to set this otherwise it returns multiple numpy arrays 
                                                                        #thinking class mode is binary.
```

Such generator can itself preprocess images, if needed, and apply augmentation to them. 

```python
img_data_gen_args = dict(rescale=1./255,
                             rotation_range=15,
                             width_shift_range=0.3,
                             height_shift_range=0.3,
                             shear_range=0.1,
                             zoom_range=0.3,
                             horizontal_flip=True,
                             vertical_flip=False,
                             fill_mode='constant')
```

Random augmentations in this case are applied every time the image is read. It means, the model during training gets slightly different images every epoch, which can significantly increase the performace and semantic segmentation results. 

### HDR

The generator above does not read HDR images saved as float32 .tif or as .npy files, so the very simple custom generator was written, which uses ***tifffile*** library for opening such files.

It only reads *batch_size* amount of images from the directory and feeds them to network. Every epoch it also shuffles the order in which the images are sent, which makes the model slightly more robust.

```python
# Attempt to create Custom Data Generator for HDR images.
# Reading .tif files from directory
# Expects both X and y folders
# TODO: add ability to open .npy ?
class CustomDataGen(keras.utils.Sequence):

    def __init__(self, X_path, y_path,
                 batch_size = 16,
                 input_size=(256, 256),
                 input_channels = 3,
                 shuffle=True):
        
        self.X_path = X_path
        self.y_path = y_path
        self.batch_size = batch_size
        self.input_size = input_size
        self.input_channels = input_channels
        self.shuffle = shuffle
        # get list of file names without extensions. names for X and y should be the same. X must have .tif extension, y .png
        self.list_IDs = [x.split('.')[0] for x in os.listdir(X_path)]
        self.total_length = len(self.list_IDs)
        self.on_epoch_end()
    
    def on_epoch_end(self):
        #         Updates indexes after each epoch
        self.indexes = np.arange(self.total_length)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, path, target_size):
        
        image_arr = np.array(image)
        
        return image_arr
    
    def __data_generation(self, list_IDs_temp):
        # Generates data containing batch_size samples # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.input_size, self.input_channels), dtype='float32')
        y = np.empty((self.batch_size, *self.input_size, 1), dtype='float32')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
          # Store sample
          X[i,] = tifImread(self.X_path + ID + '.tif')

          # Store class
          y[i,] = np.expand_dims(np.array(Image.open(self.y_path+ID+'.png')),-1)

        return X, y
    
    def __getitem__(self, index):
        # X - NumPy array of shape [batch_size, input_height, input_width, input_channel]
        # y - NumPy array of shape [batch_size, input_height, input_width]
#         'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
    def __len__(self):
        return int(np.floor(self.total_length / self.batch_size))
```

TODO: Currently it does not support augmentations, however it is in the plans, as augmentations are very important for the training process. For example changing gamma for HDR images can be crucial.

# Model

The model’s architecture and its backbone are taken from the segmentation_models library. ResNet34 with skip layers is used for faster training and experiments. The assumption is that exploding gradients with float32 0..1 values in HDR images can be a big problem. Residual networks are solving this problem the best way.

Metrics:

**Loss**: binary_crossentropy + jaccard_loss

**IOU** score

Callbacks:

For training, a callback was used that saves the model with the best IOU validation results so far after the end of an epoch.

Based on the ***keras.callbacks.ModelCheckpoint***

# Prediction

In order to make a prediction using a trained model, it is assumed that the input is also the image which was patched into correct sizes. This is done in the code, in case the image is full-sized. The model expects 4 dimensional array as an input. After prediction is made and patch masks (2-dim arrays with 1 for sky and 0 for non-sky pixels) are generated, they are “unpatchified” into original size again and saved as png file.

For now, combinations of the original images with a predicted mask are created manually. The original image and a .png mask are taken and combined with 50% opacity. However, this can be done automatically.