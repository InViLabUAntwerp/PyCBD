# Example script to create, augment and evaluate checkerboards
from PyCBD.pipelines import CBDPipeline
import matplotlib.pyplot as plt
import numpy as np
from evaluation.generate_augment import generate_CB, apply_augmentation
from evaluation.evaluation import evaluate_corners
# %%
##################
# 1) Generate
##################
image, coords, cb_shape = generate_CB(min_cb_shape=(4, 5), max_cb_shape=(13, 16), min_image_size=(640, 480),
                                      max_image_size=(2000, 2000), random_seed=True, min_cb_width=15,
                                      fixed_cb_width=None,
                                      background_image=None)
plt.imshow(image)
plt.show()

# %%
##################
# 2) Augment
##################


# for example purposed, create a random optical distortion with 1/3 probability being true
pincushion, barrel, mustache = [x if i == np.random.randint(0, 3) else False for i, x in enumerate([True, True, True])]
image, coords = apply_augmentation(image, coords, blur=1, shot_noise=0, barrel=barrel,
                                   pincushion=pincushion,
                                   mustache=mustache, n_gradient_lines=0, add_sunglints=2,
                                   invert_colors=False, translate=(0.1, 0.5), rotate=(10, 20))

plt.imshow(image)
plt.show()

# %%
##################
# 3) Evaluate
##################
detector = CBDPipeline(expand=False, predict=False)
test_res, test_board_uv, test_board_xy = detector.detect_checkerboard(image)
if test_res > 0:
    plt.imshow(image)
    plt.plot(test_board_uv[:, 0], test_board_uv[:, 1], 'ro')
    plt.title("final")
    plt.show()
    auc, correct_corners = evaluate_corners(coords, test_board_uv)
