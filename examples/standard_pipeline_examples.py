"""Standard examples for using the pipeline."""

from src.PyCBD.pipelines import CBDPipeline
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import cv2

#########
# Basic #
#########

# Load the image.
thermal_image_file = './images/thermal.tiff'
flare_image_file = './images/flare.jpg'
warped_image_file = './images/warped.jpg'
image = cv2.imread(thermal_image_file)

# Create an instance of the detector
detection_pipeline = CBDPipeline()

# Perform detection
# You can optionally give it the checkerboard size, so you'll know whether the coordinates are absolute or relative.
result, board_uv, board_xy = detection_pipeline.detect_checkerboard(image)

# Plot result
fig, ax = plt.subplots()
ax.imshow(image)
ax.plot(board_uv[:, 0], board_uv[:, 1], 'r-o', markeredgecolor='k')
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=-0.4, y=-0.20, units='inches')
ax.text(board_uv[0, 0], board_uv[0, 1], '(' + str(int(board_xy[0, 0])) + ', ' + str(int(board_xy[0, 1])) + ')',
        color="red", transform=trans_offset)
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=0.05, y=0.05, units='inches')
ax.text(board_uv[-1, 0], board_uv[-1, 1], '(' + str(int(board_xy[-1, 0])) + ', ' + str(int(board_xy[-1, 1])) + ')',
        color="red", transform=trans_offset)
plt.title("Detection result")
plt.axis('off')
plt.show()

###########################################
# Expand the board and fill in occlusions #
###########################################
# Load the image.
image = cv2.imread(flare_image_file)

# Create an instance of the detector and activate expansion and prediction.
detection_pipeline = CBDPipeline(expand=True, predict=True)

# Perform detection.
# When expanding it is recommended to give the checkerboard dimensions, so it stops when the entire board has been found.
result, board_uv, board_xy = detection_pipeline.detect_checkerboard(image, (9, 14))

# Plot result
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax.plot(board_uv[:, 0], board_uv[:, 1], 'g-o', markeredgecolor='k')
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=-0.4, y=-0.20, units='inches')
ax.text(board_uv[0, 0], board_uv[0, 1], '(' + str(int(board_xy[0, 0])) + ', ' + str(int(board_xy[0, 1])) + ')',
        color="green", transform=trans_offset)
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=0.05, y=0.05, units='inches')
ax.text(board_uv[-1, 0], board_uv[-1, 1], '(' + str(int(board_xy[-1, 0])) + ', ' + str(int(board_xy[-1, 1])) + ')',
        color="green", transform=trans_offset)
plt.title("Detection result")
plt.axis('off')
plt.show()

#################
# Heavy warping #
#################
# Load the image.
image = cv2.imread(warped_image_file)

# Create an instance of the detector and activate expansion and prediction.
detection_pipeline = CBDPipeline(expand=True, predict=True)

# Perform detection.
result, board_uv, board_xy = detection_pipeline.detect_checkerboard(image)

# Plot result
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax.plot(board_uv[:, 0], board_uv[:, 1], 'g-o', markeredgecolor='k')
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=-0.4, y=-0.20, units='inches')
ax.text(board_uv[0, 0], board_uv[0, 1], '(' + str(int(board_xy[0, 0])) + ', ' + str(int(board_xy[0, 1])) + ')',
        color="green", transform=trans_offset)
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=0.05, y=0.05, units='inches')
ax.text(board_uv[-1, 0], board_uv[-1, 1], '(' + str(int(board_xy[-1, 0])) + ', ' + str(int(board_xy[-1, 1])) + ')',
        color="green", transform=trans_offset)
plt.title("Detection result")
plt.show()

dewarped = detection_pipeline.dewarp_image(image, board_uv, board_xy)
plt.imshow(cv2.cvtColor(dewarped, cv2.COLOR_BGR2RGB))
plt.title("Dewarped image")
plt.axis('off')
plt.show()
