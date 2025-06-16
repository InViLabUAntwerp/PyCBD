"""Example for using the different components separately instead of the detection pipeline."""
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import cv2

from src.PyCBD.checkerboard_detection.checkerboard_detector import CheckerboardDetector
from src.PyCBD.checkerboard_enhancement.checkerboard_enhancer import CheckerboardEnhancer

# Load the image.
file_name = './examples/images/flare.jpg'
image = cv2.imread(file_name)

# Create a detector
checkerboard_detector = CheckerboardDetector()

# Perform detection
detected_board_uv, detected_board_xy, detected_corners_uv = checkerboard_detector.detect_checkerboard(image)

# Plot checkerboard detection results
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax.plot(detected_corners_uv[:, 0], detected_corners_uv[:, 1], 'bo', markeredgecolor='k')
ax.plot(detected_board_uv[:, 0], detected_board_uv[:, 1], 'r-o', markeredgecolor='k')
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=-0.4, y=-0.20, units='inches')
ax.text(detected_board_uv[0, 0], detected_board_uv[0, 1],
        '(' + str(int(detected_board_xy[0, 0])) + ', ' + str(int(detected_board_xy[0, 1])) + ')',
        color="red", transform=trans_offset)
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=0.05, y=0.05, units='inches')
ax.text(detected_board_uv[-1, 0], detected_board_uv[-1, 1],
        '(' + str(int(detected_board_xy[-1, 0])) + ', ' + str(int(detected_board_xy[-1, 1])) + ')',
        color="red", transform=trans_offset)
plt.title("Detected checkerboard")
plt.axis('off')
plt.show()

# Create an enhancer.
checkerboard_enhancer = CheckerboardEnhancer()

# Expand the board with the checkerboard enhancer.
expanded_board_uv, expanded_board_xy = checkerboard_enhancer.fit_and_expand_board(image,
                                                                                  detected_board_uv,
                                                                                  detected_board_xy,
                                                                                  detected_corners_uv)
# Plot expansion results
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax.plot(detected_corners_uv[:, 0], detected_corners_uv[:, 1], 'bo', markeredgecolor='k')
ax.plot(expanded_board_uv[:, 0], expanded_board_uv[:, 1], 'r-o', markeredgecolor='k')
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=-0.4, y=-0.20, units='inches')
ax.text(expanded_board_uv[0, 0], expanded_board_uv[0, 1],
        '(' + str(int(detected_board_xy[0, 0])) + ', ' + str(int(detected_board_xy[0, 1])) + ')',
        color="red", transform=trans_offset)
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=0.05, y=0.05, units='inches')
ax.text(expanded_board_uv[-1, 0], expanded_board_uv[-1, 1],
        '(' + str(int(detected_board_xy[-1, 0])) + ', ' + str(int(detected_board_xy[-1, 1])) + ')',
        color="red", transform=trans_offset)
plt.title("Expanded checkerboard")
plt.axis('off')
plt.show()

# Predict entire grid using the enhancer
predicted_board_uv, predicted_board_xy = checkerboard_enhancer.fit_and_predict_board(image, expanded_board_uv,
                                                                                     expanded_board_xy)

# Plot prediction results
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax.plot(predicted_board_uv[:, 0], predicted_board_uv[:, 1], 'g-o', markeredgecolor='k')
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=-0.4, y=-0.20, units='inches')
ax.text(predicted_board_uv[0, 0], predicted_board_uv[0, 1],
        '(' + str(int(detected_board_xy[0, 0])) + ', ' + str(int(detected_board_xy[0, 1])) + ')',
        color="green", transform=trans_offset)
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=0.05, y=0.05, units='inches')
ax.text(predicted_board_uv[-1, 0], predicted_board_uv[-1, 1],
        '(' + str(int(predicted_board_xy[-1, 0])) + ', ' + str(int(predicted_board_xy[-1, 1])) + ')',
        color="green", transform=trans_offset)
plt.title("Predicted checkerboard")
plt.axis('off')
plt.show()
