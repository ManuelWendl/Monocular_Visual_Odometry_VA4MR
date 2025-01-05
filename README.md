# Monocular Visual Odometry Pipeline - Vision Algorithms for Autonomous Robots

This repository contains a locally consistent monocular visual odometry pipeline. This pipeline is based on the lecture contents of the course "Vision Algorithms for Autonomous Robots" at ETH Zurich/UZH taught by Prof. Dr. Davide Scaramuzza in the spring autumn semester 2024/25. The pipeline estimates the camera position (vehicle pose) with respect to the camera frame of the first image from a sequence of images. 
The pipeline was tested on three different datasets, which can be accessed on the [Robot Perception Group website](http://rpg.ifi.uzh.ch/davis_data.html). The datasets are: KITTI, Parking and Malaga.

## Dependencies

The pipeline was implemented in Python 3.8 and the required packages are given in the conda environment file `environment.yml`. To create the conda environment, run the following command:

```bash
conda env create -f environment.yml
```
Using a virtual environment, the required packages are given in the `requirements.txt` file. To create the virtual environment, run the following command:

```bash
pip install -r requirements.txt
```

## Datasets

The datasets for testing the pipeline should be stored in the `data` directory. The datasets can be downloaded from the [Robot Perception Group website](http://rpg.ifi.uzh.ch/davis_data.html). The datasets are: KITTI, Parking and Malaga. The datasets should be stored in the following directory structure:

```
data/
    kitti/
        ...
    parking/
        ...
    malaga/
        ...
```

## Running the pipeline

To run the pipeline, execute the following command:

```bash
python main.py
```

The pipeline will run on the KITTI dataset by default. To run the pipeline on the Parking or Malaga dataset, change the dataset index in the `main.py` file.

```python
ds = 0  # 0: KITTI, 1: Malaga, 2: parking
```

The pipeline will output the estimated camera poses as a trajectory plot with the ground truth trajectory, extracted features and the corresponding 3D points. The plot will be saved in the 'out' directory.
```
out/interface_plot.png
```
For a continuous plotting of the estimated trajectory the according flag in the `main.py` file can be set.

```python
interface_plot = True
```


## Structure:

The VO-pipeline in 'VisualOdometry.py' is structured in:
- Initialization: Bootstrapping the pipeline with two images
    - Initial feature detection using sift
    - Initial feature matching using brute force matching and ratio test
    - Initial pose estimation using essential matrix for known intrinsics
- Continuous Operation: Estimating the camera pose for each frame
    - Feature detection 
    - Feature tracking using KLT (assuming sufficiently small frame to frame motion)
    - Triangulation of 3D points from two views
    - Pose estimation using PnP RANSAC from 3D-2D correspondences
