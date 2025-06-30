# PyCBD: Python Checkerboard Detection Toolbox

## About

Python checkerboard detection toolbox with Gaussian process based enhancement which can be used to expand detected 
checkerboards beyond occlusions, predict corners to fill in occlusions, refine corner positions, and dewarp + 
rectify the checkerboard images.

* Source: https://github.com/InViLabUAntwerp/PyCBD
* PyPi: https://pypi.org/project/PyCBD/

## Requirements

* Microsoft Windows OS
* MS VCRUNTIME14_01 needs to be installed
* Python ~=3.10, 3.11, 3.12

## Usage

Images should either be 2D grayscale (x, y) or 3D BGR (x, y, c) numpy arrays. It is recommended to use the `CBDPipeline` 
class, which combines the detector and enhancer. While it is not necessary to provide the checkerboard dimensions (the 
amount of inner corners), providing them will allow the detector to determine whether the checkerboard got detected in 
its entirety or only partially, and whether the object space coordinates are "absolute" or only "relative" to what got 
detected in the image. A simple checkerboard detection is performed as follows:

```
from PyCBD.pipelines import CBDPipeline

detector = CBDPipeline()
result, board_uv, board_xy = detector.detect_checkerboard(image)
```

The enhancer that handles board expansion and prediction is not used by default and is activated by passing additional
arguments to the pipeline constructor. When using board expansion, it is recommended to provide the checkerboard
dimensions because it is used to stops/skip expansion when the entire board has been found. In order for the enhancer
to work properly, the detected corners must have the correct coordinates, otherwise all results achieved with the
enhancer will be wrong. The enhancer is activated as follows:

```
from PyCBD.pipelines import CBDPipeline

detector = CBDPipeline(expand=True, predict=True)
result, board_uv, board_xy = detector.detect_checkerboard(image, (n_rows, n_cols))
```

It is also possible to use another detector in combination with the pipeline. The requirements are that this detector 
is contained within a class that has a  `detect_checkerboard` method that accepts the same inputs and provides the 
same outputs as our `CheckerboardDetector` class. A class instance of the detector can then be passed to the 
`CBDPipeline` constructor:

```
from PyCBD.pipelines import CBDPipeline
import YourCustomDetector

detector = CBDPipeline(YourCustomDetector())
result, board_uv, board_xy = detector.detect_checkerboard(image)
```

Instead of using the pipeline users can also use the separate `CheckerboardDetector` for detection:

```
from PyCBD.checkerboard_detection.checkerboard_detector import CheckerboardDetector
from PyCBD.pipelines import prepare_image


prepared_image = prepare_image(image)
checkerboard_detector = CheckerboardDetector()
detected_board_uv, detected_board_xy, detected_corners_uv = checkerboard_detector.detect_checkerboard(prepared_image)
```

and the `CheckerboardEnhancer` for expanding the board and predicting corners:

```
from PyCBD.checkerboard_enhancement.checkerboard_enhancer import CheckerboardEnhancer

checkerboard_enhancer = CheckerboardEnhancer()
expanded_board_uv, expanded_board_xy = checkerboard_enhancer.fit_and_expand_board(image,
                                                                                  detected_board_uv,
                                                                                  detected_board_xy,
                                                                                  detected_corners_uv)
predicted_board_uv, predicted_board_xy = checkerboard_enhancer.fit_and_predict_board(image, 
                                                                                     expanded_board_uv,
                                                                                     expanded_board_xy)
```

Finally, the enhancer can be used to remove warping and perspective error from the image after the checkerboard has been
detected. Both `CBDPipeline` and `CheckerboardEnhancer` have a `dewarp_image` method for this purpose:

```
dewarped = checkerboard_enhancer.dewarp_image(image, board_uv, board_xy)
```

In case the detection fails, or you get a weird outcome, you can set certain flags on the different classes to show 
intermediate results and diagnose the problem, and configure the package logger, so you get additional info prints 
during execution. If there are problems at the enhancer level, it is possible they can be resolved by adjusting the 
parameters. Please refer to the documentation for additional in-depth information.

## Citation

    @Article{math11224568,
        AUTHOR = {Hillen, Michaël and De Boi, Ivan and De Kerf, Thomas and Sels, Seppe and Cardenas De La Hoz, Edgar and Gladines, Jona and Steenackers, Gunther and Penne, Rudi and Vanlanduit, Steve},
        TITLE = {Enhanced Checkerboard Detection Using Gaussian Processes},
        JOURNAL = {Mathematics},
        VOLUME = {11},
        YEAR = {2023},
        NUMBER = {22},
        ARTICLE-NUMBER = {4568},
        URL = {https://www.mdpi.com/2227-7390/11/22/4568},
        ISSN = {2227-7390},
        DOI = {10.3390/math11224568}
    }

## License

Distributed under the GNU General Public License v3.0. Check the `LICENCE` files for more info.

## Contact

InViLab - [invilab@uantwerpen.be](mailto:invilab@uantwerpen.be) - [website](https://www.invilab.be/) - 
[LinkedIn](https://www.linkedin.com/company/invilab-uantwerp)

## Acknowledgements

The checkerboard detector in this toolbox is a modified version of the C++ implementation of libcbdetect 
[[1]](#ref1)[[2]](#ref2). For the Gaussian processes we use the GPy library [[3]](#ref3)

## References

<a id="ref1">[1]</a> 
Geiger, A., Moosmann, F., Car, Ö., & Schuster, B. (2012, May). Automatic camera and range sensor calibration using a 
single shot. In Robotics and Automation (ICRA), 2012 IEEE International Conference on (pp. 3936-3943). IEEE.

<a id="ref2">[2]</a> 
ftdlyc (March 13 2020). Unofficial implemention of libcbdetect in C++. [https://github.com/ftdlyc/libcbdetect](https://github.com/ftdlyc/libcbdetect)

<a id="ref3">[3]</a> 
GPy (since 2012). GPy: A Gaussian process framework in python. [http://github.com/SheffieldML/GPy](http://github.com/SheffieldML/GPy)
