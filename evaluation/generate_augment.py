import numpy as np
from PIL import Image, ImageDraw
import albumentations as A
import cv2


def generate_CB(min_cb_shape=(4, 5), max_cb_shape=(13, 16), min_image_size=(640, 480), max_image_size=(2000, 2000),
                random_seed=False, min_cb_width=45, fixed_cb_width=None, background_image=None):
    """
    Generates a checkerboard image and the coordinates of the non-border checkerboard corners.
    :param min_cb_shape: Tuple of integers representing the minimum number of rows and columns of the checkerboard.
    :param max_cb_shape: Tuple of integers representing the maximum number of rows and columns of the checkerboard.
    :param min_image_size: Tuple of integers representing the minimum size of the image.
    :param max_image_size: Tuple of integers representing the maximum size of the image.
    :param random_seed: Seed for the random number generator.
    :param min_cb_width: Minimum width of the checkerboard squares.
    :param fixed_cb_width: The fixed width of the checkerboard squares.
    :param background_image: The path to the background image to be used.
    :return: Tuple of numpy arrays, the first being the checkerboard image,
    the second being the coordinates of the checkerboard corners and the third the checkerboard shape
    """
    if not random_seed:
        np.random.seed(random_seed)

    # Random Generate the CB size and make sure they are not equal
    if min_cb_shape == max_cb_shape:
        cb_shape = (min_cb_shape)
    else:
        while True:
            cb_shape = (
                np.random.randint(min_cb_shape[0], max_cb_shape[0]),
                np.random.randint(min_cb_shape[1], max_cb_shape[1]))
            if cb_shape[0] != cb_shape[1]:
                break

    # Add 1 to because when defining the cb, we only count the inner points.
    # cb_shape = (cb_shape[0] + 1, cb_shape[1] + 1)

    if min_image_size == max_image_size:
        image_size = min_image_size
    else:
        image_size = (
            np.random.randint(min_image_size[0], max_image_size[0]),
            np.random.randint(min_image_size[1], max_image_size[1]))

    # try to get background image
    if background_image is not None:
        image = Image.open(background_image)
        image = image.resize(image_size)

    else:
        randcolor = list(np.random.choice(range(256), size=3))
        image = Image.new("RGB", image_size, tuple(randcolor))

    # calculate the max value that the checkerboard can be. The -5 is to make sure that the CB has a 5 pixel margin in the image.
    cb_width_max = min([(image_size[0] - 5) // (cb_shape[0] + 1), (image_size[1] - 5) // (cb_shape[1] + 1)])
    if fixed_cb_width is None:
        cb_width = np.random.randint(min_cb_width, cb_width_max)
    else:
        if cb_width_max < fixed_cb_width:
            print(
                'unable to set the chosen fixed checkerboard size due to shape constraints. Choosing a random cb width.')
            cb_width = np.random.randint(min_cb_width, cb_width_max)
        else:
            cb_width = fixed_cb_width

    draw = ImageDraw.Draw(image)
    random_white = np.random.randint(230, 255)
    random_black = np.random.randint(0, 10)
    # create random start positions
    x = np.random.randint(1, image_size[0] - 5 - cb_width * (cb_shape[0] + 1))
    y = np.random.randint(1, image_size[1] - 5 - cb_width * (cb_shape[1] + 1))
    coords = []
    for i in range(0, cb_shape[1] + 1):
        for j in range(0, cb_shape[0] + 1):
            x1 = x + j * cb_width
            y1 = y + i * cb_width
            x2 = x1 + cb_width
            y2 = y1 + cb_width
            color = (random_black, random_black, random_black) if (i + j) % 2 == 0 else (
                random_white, random_white, random_white)
            draw.rectangle((x1, y1, x2, y2), fill=color)
            if j > 0 and i > 0:
                coords.append([x1 - 0.5, y1 - 0.5])
    coords = np.array(coords)

    # Add border, otherwise opencv does not work at all
    top_left_x, top_left_y = coords[0] - cb_width - 4
    bottom_right_x, bottom_right_y = coords[-1] + cb_width + 4
    draw.rectangle((top_left_x, top_left_y, bottom_right_x, bottom_right_y), width=5,
                   outline=(random_white, random_white, random_white))

    image = np.array(image)

    return image, coords, cb_shape


def apply_augmentation_kp_inside(image, keypoints, blur=3, shot_noise=2, scale=(1, 1), translate=(0, 0),
                                 perspective=(0, 0), rotate=(0, 0), pincushion=False, barrel=False, mustache=False,
                                 n_gradient_lines=False, add_sunglints=False, invert_colors=False, shear=0):
    kp_out_image = True
    i = 0
    while kp_out_image == True:
        output_image, output_keypoints = apply_augmentation(image, keypoints, blur=blur,
                                                            shot_noise=shot_noise,
                                                            scale=scale, translate=translate,
                                                            perspective=perspective, rotate=rotate,
                                                            pincushion=pincushion,
                                                            barrel=barrel, mustache=mustache,
                                                            n_gradient_lines=n_gradient_lines,
                                                            add_sunglints=add_sunglints,
                                                            invert_colors=invert_colors)
        try:
            kp_out_image = np.any(output_keypoints[:, 0] < 0) or np.any(
                output_keypoints[:, 0] >= output_image.shape[1]) or np.any(
                output_keypoints[:, 1] < 0) or np.any(output_keypoints[:, 1] >= output_image.shape[0])
        except:
            pass
        i += 1
        if i > 100:
            print('WARNING: Could not find a valid augmentation after 100 tries. Returning original image.')
            return image, keypoints

    return output_image, output_keypoints


def apply_gradient_line(image, coords):
    import random
    import cv2
    img = image

    # generate random thickness
    min_thickness = np.abs(coords[0][0] - coords[1][0])
    max_thickness = np.abs(coords[0][0] - coords[2][0])
    w = np.random.randint(min_thickness, max_thickness)

    # Construct center line
    height, width, _ = img.shape
    line = np.zeros_like(img)
    x = int(np.max(coords[:, 0]) - np.min(coords[:, 0]))
    line[:, (x - w):(x + w + 1)] = (0, 0, 0)

    # Construct line alpha mask
    alpha = np.zeros_like(img, np.float32)
    alpha[:, (x - w):x, :] = np.repeat(np.arange(1, w + 1)[:, np.newaxis] / (w + 1), 3, axis=1)
    alpha[:, x:(x + w + 1), :] = np.repeat(np.arange(w + 1, 0, -1)[:, np.newaxis] / (w + 1), 3, axis=1)

    # Generate random horizontal and vertical shifts
    h_shift = np.random.randint(-width // 4, width // 4)
    v_shift = np.random.randint(-height // 4, height // 4)

    # Generate a random rotation angle
    angle = np.random.randint(-45, 45)

    # Perform the shift and rotation
    M = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
    M[0, 2] += h_shift
    M[1, 2] += v_shift
    alpha = cv2.warpAffine(alpha, M, (width, height))
    line = cv2.warpAffine(line, M, (width, height))

    # Blend into img
    img = ((1 - alpha) * img + alpha * line).astype(img.dtype)

    return img


def apply_augmentation(image, keypoints, blur=3, shot_noise=2, scale=(1, 1), translate=(0, 0),
                       perspective=(0, 0), rotate=(0, 0), pincushion=False, barrel=False, mustache=False,
                       n_gradient_lines=0, add_sunglints=False, invert_colors=False, shear=0):
    """
    Applies a series of augmentations to an image and its keypoints. The augmentations include gaussian blur, shot noise,
    affine transformation, perspective transformation, and optical distortions (pincushion, barrel, mustache).

    Parameters:
        image (np.ndarray): The input image as a numpy array.
        keypoints (np.ndarray): The keypoints associated with the input image as a numpy array.
        blur (int): The kernel size for blur. Default is 3.
        shot_noise (int): The multiplier for shot noise. Default is 2.
        scale (tuple): The scaling factor for the affine transformation as a tuple of floats. Default is (1,1).
        translate (tuple): The translation factor for the affine transformation as a tuple of floats. Default is (0,0).
        perspective (tuple): The scaling factor for the perspective transformation as a tuple of floats. Default is (0,0).
        rotate (tuple): The rotation factor for the affine transformation as a tuple of floats. Default is (0,0).
        pincushion (bool): Whether or not to apply pincushion distortion. Default is False.
        barrel (bool): Whether or not to apply barrel distortion. Default is False.
        mustache (bool): Whether or not to apply mustache distortion. Default is False.
        invert_colors (bool): Whether or not to invert the colors. Default is False.
        n_gradient_lines (int): the number of gradient lines added to image. Default is 0.
    Returns:
        tuple: A tuple containing the augmented image and the augmented keypoints.
    """
    coords_norm_x = keypoints[:, 0] / image.shape[1]
    coords_norm_y = keypoints[:, 1] / image.shape[0]

    # Define transform pipeline
    augmentations = [A.Blur(blur_limit=(blur, blur), always_apply=True,p = 1),
                     # A.MultiplicativeNoise(multiplier=(1 - shot_noise, 1 + shot_noise), elementwise=True,
                     #                       per_channel=True, always_apply=True),
                     A.Affine(scale=scale, translate_percent=translate, rotate=5, fit_output=True, always_apply=True,
                              p=1),
                     A.Perspective(scale=perspective, fit_output=True, p=1, keep_size=True, pad_mode=1,
                                   always_apply=True),
                     ]

    transform = A.Compose(augmentations, keypoint_params=A.KeypointParams(format='xy'))
    # Apply transform to image and keypoints
    data = transform(image=image, keypoints=keypoints)
    image = data["image"]
    keypoints = np.array(data["keypoints"])
    if shot_noise:
        shape = image.shape
        lF = shot_noise *0.1
        hF = shot_noise *0.1
        noise = np.zeros(shape) * 1.
        noise = noise + np.random.uniform(-lF, lF)
        noise = noise + np.random.uniform(-1, 1, (shape)) * np.random.uniform(-hF, hF) * 255
        image = image + noise.astype('uint8')

    if add_sunglints:
        for glint_number in range(add_sunglints):
            # pick a random keypoint
            random_center = keypoints[np.random.randint(0, len(keypoints))]
            # get the width of the keypoint
            width = int((keypoints[1][0] - keypoints[0][0]) // 2)
            cv2.circle(img=image, center=random_center.astype(int), radius=width, color=(255, 255, 255), thickness=-1)

    # add gradient lines
    for line in range(n_gradient_lines):
        image = apply_gradient_line(image, keypoints)

    # Invert colors
    if invert_colors:
        image = cv2.bitwise_not(image)

    # NOTE: Because Albumentations does not support keypoint distortion for optical distortions, the code below is
    # used to create pincushion, barrel and mustache distortion
    # Apply optical distortions if any of the flags (pincushion, barrel, mustache) is set to True
    cb_in_image = False
    check_loop = 0
    if any([mustache, barrel, pincushion]):
        while not cb_in_image:
            check_loop += 1
            if check_loop > 1:
                print(check_loop)
            if mustache:
                d_coef = (-np.random.uniform(0, 2), 0, 0, 0, np.random.randint(1, 10))
            elif pincushion:
                d_coef = (np.random.randint(1, 10), 0, 0, 0)
            elif barrel:
                d_coef = (-np.random.uniform(0, 1.5), 0, 0, 0)
            h, w = image.shape[:2]
            # compute its diagonal
            f = (h ** 2 + w ** 2) ** 0.5
            # set the image projective to carrtesian dimension
            K = np.array([[f, 0, w / 2],
                          [0, f, h / 2],
                          [0, 0, 1]])

            # Generate new camera matrix from parameters
            M, n = cv2.getOptimalNewCameraMatrix(K, d_coef, (w, h), 0)
            # Generate look-up tables for remapping the camera image
            remap = cv2.initUndistortRectifyMap(K, d_coef, None, M, (w, h), 5)

            # Remap the original image to a new image
            output_image = cv2.remap(image, *remap, cv2.INTER_LINEAR)

            if keypoints is not None:
                dst = cv2.undistortPoints(keypoints[:, None, :], K, d_coef, None, M)
                dst = np.squeeze(dst)

                return output_image, dst
            else:
                return output_image

    else:
        return image, np.array(data["keypoints"])


def reshape_coordinates(coords, cb_shape):
    """
    Reshapes and transposes the coordinates going from left-right and top-bottom to top-bottom and left-right.

    Parameters:
    coords (ndarray): Original coordinates of the grid.
    cb_shape (tuple): Tuple containing the shape of the grid in the form (rows, cols).

    Returns:
    ndarray: Reshaped and transposed coordinates of the grid.
    """
    # reshape the coordinates to have shape (cols, rows, 2)
    coords = coords.reshape(cb_shape[1], cb_shape[0], 2)
    # transpose the x-coordinates and y-coordinates and stack them together along the 3rd axis
    coords_transposed = np.stack([coords[:, :, 0].T, coords[:, :, 1].T], axis=2)
    # reshape the transposed coordinates to have shape (rows*cols, 2)
    coords_transposed = coords_transposed.reshape(cb_shape[1] * cb_shape[0], 2)
    return coords_transposed


if __name__ == 'main':
    import matplotlib.pyplot as plt

    min_cb_shape = (4, 5)
    max_cb_shape = (4, 5)
    min_image_size = (640, 480)
    max_image_size = (2000, 2000)
    random_seed = False
    min_cb_width = 15
    fixed_cb_width = None
    background_image = None

    image, coords = generate_CB(min_cb_shape=(4, 5), max_cb_shape=(13, 16), min_image_size=(640, 480),
                                max_image_size=(2000, 2000), random_seed=False, min_cb_width=15, fixed_cb_width=None,
                                background_image=None)

    image, coords = apply_augmentation(image, coords, blur=7, shot_noise=4, scale=(1, 0.5),
                                       perspective=(0.05, 0.6), rotate=(10, 20))

    plt.figure()
    plt.imshow(image)
    plt.plot(coords[:, 0], coords[:, 1], 'rx')
