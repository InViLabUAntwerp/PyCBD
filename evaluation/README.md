# Checkerboard Image Generator
This code provides functions to 
* Generate a checkerboard image and corners. 
* Augment the image and the corners.
* Evaluate the detection methods.

**Examples can be found in examply.py** 

## Usage
### 1) Generate checkerboard and keypoints
The generate_CB() function takes in the following parameters:

* **min_cb_shape** (tuple): Tuple of integers representing the minimum number of rows and columns of the checkerboard.  
* **max_cb_shape** (tuple): Tuple of integers representing the maximum number of rows and columns of the checkerboard.  
* **min_image_size** (tuple): Tuple of integers representing the minimum size of the image.  
* **max_image_size** (tuple): Tuple of integers representing the maximum size of the image.  
* **random_seed** (int): Seed for the random number generator.  
* **min_cb_width** (int): Minimum width of the checkerboard squares.  
* **fixed_cb_width** (int): The fixed width of the checkerboard squares.  
* **background_image** (str): The path to the background image to be used.  

**Example**
```python
from Evaluation.CreateDataset import generate_CB

image, coords = generate_CB(min_cb_shape=(4, 5), max_cb_shape=(13, 16), min_image_size=(640, 480),
                            max_image_size=(2000, 2000), random_seed=False, min_cb_width=15, fixed_cb_width=None,
                            background_image=None)
```
### 2) Augment image and keypoints
The _apply_augmentation()_ function takes in the following parameters:

* **image** (numpy.ndarray): The image to be augmented.  
* **keypoints** (numpy.ndarray): The keypoints of the image.  
* **gaussian_blur** (float): The amount of Gaussian blur to be applied to the image.  
* **shot_noise** (float): The amount of shot noise to be applied to the image.  
* **scale** (tuple): The scale factor to be applied to the image.  
* **translate** (tuple): The translation factor to be applied to the image.  
* **perspective** (tuple): The perspective factor to be applied to the image.  
* **rotate** (tuple): The rotation angle to be applied to the image.  
* **pincushion** (bool): Whether to apply pincushion distortion to the image.  
* **barrel** (bool): Whether to apply barrel distortion to the image.  
* **mustache** (bool): Whether to apply mustache distortion to the image.  

The _apply_augmentation_kp_inside()_ takes in the same arguments but includes an extra check to see if all the checkpoints are inside the actual image  
If the parameters are left blank, the distortions will execute in a random fashion. This does not include the
pincushion, barrel and mustache distortions. However you can use following function to randomize these distortions:
```python
randint = np.random.randint(0,3)
pincushion,barrel, mustache = [x if i==randint[0] else False for i,x in enumerate([True, True, True])]
```

**Example**
```python
from Evaluation.CreateDataset import apply_augmentation

image, coords = apply_augmentation(image, coords, gaussian_blur=7, shot_noise=4, scale=(1, 0.5),
                                   perspective=(0.05, 0.6), pincushion=True)
```
