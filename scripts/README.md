# Robot Arm Solve Sudoku

For this project, we programmed a robot arm Edwin to solve a 4x4 Sudoku.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
YOu can download the sudoku grid used for this algorithm [here](https://drive.google.com/drive/folders/1KD8HssK76iSOaB6Fd9mHtr5TbYzTBgDm?usp=sharing)


To get the training data for different fonts, you can download the font dataset from [this link](https://drive.google.com/drive/folders/1nlYUBKpYsFesmsZJGm-G84Y8ue_JjK9_?usp=sharing)

To generate font images for training, run the following code:
```
python /scripts/training_data/font_dataset_generator.py
```
The images will be stored in /training_data/font_images
### Running the code

Set up Ros Environment


```
roscore
```
Connect to Edwin node

```
rosrun irl edwin_node.py
```

Set up edwin routes and behaviors
```
rosrun irl edwin_routes.py
rosrun irl edwin_behaviors.py
```

Connect to camera
```
rosrun usb_cam usb_cam_node _video_device:='/dev/video1'
```

Set up edwin write node
```
rosrun irl edwin_write.py
```

Main class of Sudoku game
```
python sudoku_main.py
```

## Built With
* Ros
* OpenCV
* KNN


## Authors

* **Khang Vu**
* **Nick Steelman**  
* **Sherrie Shen**

## Acknowledgments

* ![Simple Digit Recognition OCR in OpenCV-Python](https://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python)
