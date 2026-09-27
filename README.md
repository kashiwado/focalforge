# FocalForge

FocalForce is a simple OpenCV based Python application that allows a user to
either start from scratch or load pre-existing Kannala-Brandt camera calibration
parameters to allow field robot or vehicle calibration parameter adjustments
without the need for tedious checkerboard plates and offline data processing.

This is particularly useful if only a minor change has occurred in a previously
determined calibration. FocalForge is also useful for taking a brute-force approach
to determining an approximate camera calibration for a camera with unknown lens
intrinsics.

The application is used by passing a video file or video device (like a webcam)
as a command-line parameter. The video feed functions as the backdrop for a 
3D->2D grid projection visualization, allowing approximate visual verification
of lens projection geometry and manual verification of the extrinsic and intrinsic
calibration via e.g. a tape measure.

## Usage

Keyboard interface:
```
  Main menu:
    A / D      - switch between Intrinsic / Extrinsic menus
    Enter      - enter the highlighted menu
    Q / ESC    - quit (ESC also goes back from sub-menus)

  Intrinsic / Extrinsic sub-menus:
    W / S      - move between parameters (non-edit mode)
                 OR adjust the selected parameter value (edit mode)
    Enter      - toggle edit mode (activate parameter / confirm value)
    A / ESC    - back to main menu
```

## Dependencies

* OpenCV 4
* numpy 


