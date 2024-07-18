# Autonomous Car Line Follower Using Q-Learning

![Line Follower](https://user-images.githubusercontent.com/90126322/219383180-47eb70ba-f8ad-4424-ab5c-274ea07d8fed.gif)

## Overview

This project implements the Q-learning algorithm for an autonomous car to follow a line. It consists of a Simulation Environment, a Real-World Environment, and Image Processing components.

## Simulation Environment

The **Simulation Environment** is where the robot learns to follow the line using Q-learning. It simulates various scenarios to optimize path following.

## Real-World Environment

The **Real-World Environment** deploys the trained model onto a physical robot, demonstrating its line-following abilities in real-world conditions.

## Image Processing

**Image Processing** involves capturing and processing images from a camera (Logitech C270 for real-world and PyBullet library for simulation). Images are resized to 32x32 pixels, converted to binary format, and filtered to detect the red tape line. These processed images serve as input for the Q-learning algorithm to determine optimal actions for line following.

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



