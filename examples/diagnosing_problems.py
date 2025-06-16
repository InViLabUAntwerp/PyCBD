"""
This is an example scrip for enabling options that allow you to diagnose potential problems with the expansion and/or
detection.
"""

from src.PyCBD.pipelines import CBDPipeline
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import cv2
import logging
from src.PyCBD.logger_configuration import configure_logger


# Configure the package loggers so info messages get printed in the console
configure_logger(level=logging.INFO)

# Load the image.
image_file = './examples/images/broken.jpg'
image = cv2.imread(image_file)

# Pass additional arguments to constructor to activate expansion and prediction.
detection_pipeline = CBDPipeline(expand=True, predict=True)
# The issue can be resolved by increasing the max_expansion_factor. Uncomment the next line to resolve the issue.
detection_pipeline.max_expansion_factor = 3

# Enable diagnostic plots.
detection_pipeline.plot_pipeline_steps = True  # intermediate results between each large step.
detection_pipeline.must_plot_iterations = True  # intermediate results during expansion

# Perform detection
result, board_uv, board_xy = detection_pipeline.detect_checkerboard(image, (9, 6))

#Plot results
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
