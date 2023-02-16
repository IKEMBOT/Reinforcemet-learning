# Line-Follower-using-Q-Learning

# Description

The main of the project is Q-learning (Reinforced Learning) algorithm. There are three main components to the system: The Simulation Environment, the Real-World Environment, and the Image Processor. The Simulation environment is used to robot simulation to train the robot was able to follow the line based on Q-learning Method. The Real-World Environment to execute the final result after simulation. and The Image Processor takes in images camera(Logitech C270) for the real-world case and the simulation case uses a camera provided by pybullet function, resizes them into a 32x32 image, and then converts them into a binary image that filters specifically for the red tape line. Each image is an observation to define the state through the Q-Learning algorithm to generate probability values for each possible action. lastly, after many iterations, the probability space becomes optimized to follow the line.

# Requirement
The requirements.txt file for installation of some packages this project using:

    pip install -r requirements.txt


# Videos
    Simulation Video :
  ![ezgif com-video-to-gif(2)](https://user-images.githubusercontent.com/90126322/219383180-47eb70ba-f8ad-4424-ab5c-274ea07d8fed.gif


    Demonstration Robot in Real-World using Jetson-Nano :
  ![ezgif com-video-to-gif(1)](https://user-images.githubusercontent.com/90126322/219380977-fce36e4a-49b8-4ff9-8f52-dfa2f12ae330.gif)


