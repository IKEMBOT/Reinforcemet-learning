# REINFORCEMENT LEARNING PROJECT
## Line Follower using Q-Learning

![Line Follower](https://user-images.githubusercontent.com/90126322/219383180-47eb70ba-f8ad-4424-ab5c-274ea07d8fed.gif)

## Description

This project focuses on implementing the Q-learning (Reinforcement Learning) algorithm for a line-following robot. The system comprises three main components: the Simulation Environment, the Real-World Environment, and the Image Processing.

### Simulation Environment
The **Simulation Environment** is used to train the robot to follow the line based on the Q-learning method. It involves running simulations to enable the robot to learn and optimize its path following the line.

### Real-World Environment
The **Real-World Environment** is utilized to execute the final results after simulation. The trained robot is deployed in the real world to demonstrate its line-following capabilities.

### Image Processing
The **Image Processing** takes images from a camera (Logitech C270) in the real-world case and uses a camera provided by the PyBullet library in the simulation case. The images are resized into 32x32 images and then converted into binary images, specifically filtering for the red tape line. Each image serves as an observation, defining the state, and the Q-learning algorithm generates probability values for each possible action. After multiple iterations, the probability space becomes optimized to enable the robot to follow the line effectively.

## Requirements

The project requires the installation of some packages, which are specified in the `requirements.txt` file. To install these packages, run the following command:

    pip install -r requirements.txt



## Videos

### Simulation Video:
Below is a video demonstrating the line-following behavior in the simulation environment.

![Simulation Video](https://user-images.githubusercontent.com/90126322/219383180-47eb70ba-f8ad-4424-ab5c-274ea07d8fed.gif)

### Demonstration Robot in Real-World using Jetson-Nano:
Below is a video showing the robot's line-following performance in the real world using a Jetson-Nano board.

![Real-World Video](https://user-images.githubusercontent.com/90126322/219380977-fce36e4a-49b8-4ff9-8f52-dfa2f12ae330.gif)



