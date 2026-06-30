# PyCBD: Python Checkerboard Detection Toolbox

## About

PyCBD is a Python toolbox for checkerboard detection with optional Gaussian process-based enhancement. It can be used to

- detect checkerboards in grayscale or color images,
- expand partially detected checkerboards beyond occlusions,
- predict missing corners,
- refine corner locations, and
- dewarp and rectify checkerboard images.

The original Gaussian process implementation described in our publication was based on **GPy**. Since version 1.5, PyCBD uses **GPyTorch** as its Gaussian process backend, providing improved performance and GPU acceleration through PyTorch.

- **Source:** https://github.com/InViLabUAntwerp/PyCBD
- **PyPI:** https://pypi.org/project/PyCBD/

---

## What's New in v1.5

### GPyTorch backend

PyCBD v1.5 replaces the original **GPy**-based Gaussian process implementation with **GPyTorch**.

This change was made because GPy is no longer actively maintained and is not compatible with the NumPy 2.x ecosystem, making installation increasingly difficult on modern Python environments. GPyTorch is actively maintained, built on PyTorch, and provides a modern, high-performance Gaussian process framework with optional CUDA acceleration.

The underlying Gaussian process methodology remains unchanged and is still based on the approach presented in our publication. This update only replaces the implementation backend to improve long-term maintainability, compatibility, and performance.

**Note:** The original publication accompanying PyCBD used GPy for its Gaussian process implementation. We continue to acknowledge and credit GPy for its role in the original research, while current versions of PyCBD use GPyTorch for all Gaussian process functionality.

---

## Requirements

- Microsoft Windows
- Microsoft Visual C++ Runtime (VCRUNTIME14_01)
- Python 3.10–3.13

### Optional GPU acceleration

The Gaussian process enhancement module supports CUDA acceleration through **PyTorch** when a CUDA-enabled PyTorch installation is available. If no compatible GPU is detected (or a CPU-only version of PyTorch is installed), PyCBD automatically falls back to CPU execution.

See the official PyTorch installation instructions to install the appropriate CUDA-enabled version for your system:

https://pytorch.org/get-started/locally/

---

## Usage

Images should be provided as NumPy arrays in one of the following formats:

- grayscale: `(height, width)`
- BGR color: `(height, width, channels)`

The recommended entry point is the `CBDPipeline`, which combines checkerboard detection and optional Gaussian process enhancement.

Providing the checkerboard dimensions (number of inner corners) is optional but recommended, as it allows PyCBD to determine whether the full checkerboard has been detected and to assign absolute object-space coordinates.

### Basic checkerboard detection

```python
from PyCBD.pipelines import CBDPipeline

detector = CBDPipeline()
result, board_uv, board_xy = detector.detect_checkerboard(image)
```

---

### Detection with Gaussian process enhancement

The enhancement module is disabled by default. Enable it by setting `expand=True` and/or `predict=True`.

Providing the checkerboard dimensions is recommended when using board expansion so the algorithm knows when the complete board has been recovered.

```python
from PyCBD.pipelines import CBDPipeline

detector = CBDPipeline(expand=True, predict=True)
result, board_uv, board_xy = detector.detect_checkerboard(image, (n_rows, n_cols))
```

---

### Using a custom detector

Any detector implementing a

```python
detect_checkerboard(image, dimensions=None)
```

method with the same input/output interface as `CheckerboardDetector` can be used with the pipeline.

```python
from PyCBD.pipelines import CBDPipeline
import YourCustomDetector

detector = CBDPipeline(YourCustomDetector())
result, board_uv, board_xy = detector.detect_checkerboard(image)
```

---

### Using the detector directly

```python
from PyCBD.checkerboard_detection.checkerboard_detector import CheckerboardDetector
from PyCBD.pipelines import prepare_image

prepared_image = prepare_image(image)

checkerboard_detector = CheckerboardDetector()
detected_board_uv, detected_board_xy, detected_corners_uv = (
    checkerboard_detector.detect_checkerboard(prepared_image)
)
```

---

### Using the enhancer directly

```python
from PyCBD.checkerboard_enhancement.checkerboard_enhancer import CheckerboardEnhancer

checkerboard_enhancer = CheckerboardEnhancer()

expanded_board_uv, expanded_board_xy = (
    checkerboard_enhancer.fit_and_expand_board(
        image,
        detected_board_uv,
        detected_board_xy,
        detected_corners_uv,
    )
)

predicted_board_uv, predicted_board_xy = (
    checkerboard_enhancer.fit_and_predict_board(
        image,
        expanded_board_uv,
        expanded_board_xy,
    )
)
```

---

### Dewarping an image

After successful detection, both `CBDPipeline` and `CheckerboardEnhancer` can remove perspective distortion and board warping.

```python
dewarped = checkerboard_enhancer.dewarp_image(image, board_uv, board_xy)
```

---

## Debugging

If detection fails or produces unexpected results, several debugging options are available:

- enable visualization flags to inspect intermediate processing steps,
- configure the package logger for additional runtime information,
- adjust the Gaussian process enhancement parameters if expansion or prediction behaves unexpectedly.

For detailed explanations of all available options, please refer to the documentation.

---

## Citation

If you use PyCBD in academic work, please cite:

```bibtex
@Article{math11224568,
    AUTHOR = {Hillen, Michaël and De Boi, Ivan and De Kerf, Thomas and Sels, Seppe and Cardenas De La Hoz, Edgar and Gladines, Jona and Steenackers, Gunther and Penne, Rudi and Vanlanduit, Steve},
    TITLE = {Enhanced Checkerboard Detection Using Gaussian Processes},
    JOURNAL = {Mathematics},
    VOLUME = {11},
    YEAR = {2023},
    NUMBER = {22},
    ARTICLE-NUMBER = {4568},
    DOI = {10.3390/math11224568}
}
```

---

## License

PyCBD is distributed under the GNU General Public License v3.0. See the `LICENSE` file for details.

---

## Contact

**InViLab — University of Antwerp**

Email: invilab@uantwerpen.be

Website: https://www.invilab.be/

LinkedIn: https://www.linkedin.com/company/invilab-uantwerp

---

## Acknowledgements

The checkerboard detector included in PyCBD is based on a modified version of the C++ implementation of **libcbdetect** [[1]](#ref1), itself derived from the work of Geiger *et al.* [[2]](#ref2).

The Gaussian process methodology presented in the accompanying publication was originally implemented using **GPy** [[3]](#ref3). Beginning with **PyCBD v1.5**, the implementation has been migrated to **GPyTorch** [[4]](#ref4), which provides modern PyTorch integration and optional GPU acceleration while preserving the methodology described in the original work.

---

## References

<a id="ref1">[1]</a>  
ftdlyc. *Unofficial implementation of libcbdetect in C++.*  
https://github.com/ftdlyc/libcbdetect

<a id="ref2">[2]</a>  
Geiger, A., Moosmann, F., Car, Ö., & Schuster, B. (2012). *Automatic camera and range sensor calibration using a single shot.* Proceedings of the IEEE International Conference on Robotics and Automation (ICRA).

<a id="ref3">[3]</a>  
The GPy Authors. *GPy: A Gaussian Process Framework in Python.*  
https://github.com/SheffieldML/GPy

<a id="ref4">[4]</a>  
Gardner, J. R., Pleiss, G., Bindel, D., Weinberger, K. Q., & Wilson, A. G. (2018). *GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration.* Advances in Neural Information Processing Systems (NeurIPS 31).