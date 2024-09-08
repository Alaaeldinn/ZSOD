# ZSOD - Zero Shot Object Detection 

## Overview

Zero-shot Object detection to enhance the performance and accuracy of detection and recognation applications. This methodology expands the capabilities of traditional object detection systems by allowing them to recognize and classify objects even without specific training examples.

the code is for just simple survey on using clip and owl-vit in zero shot object detection task. 

<p align="center">
    <img src="z-media/outputs1.png" width="500" alt="Image 1">
    <img src="z-media/OUT2.png" width="300" alt="Image 2">
</p>

## Brainstorm
- comparing between Clip and Owl-vit in objective of Zero shot object detection
- Tradeoff : while comparing between the two methods , accuarcy could be tradeoff , model size , latency, etc ...
- but about using embbeding in Detection Task  using image-language , image-image embeddings would be interesting topic cause of how we descripe objects
- also there are other methodolgies like yolo-world could be found but the choice of these two methods was considred , was that clip was one of the most cited papers specially in the embbeddings topics, the owl-vit one of interesting models by google tradeoff between latecny , accuarcy developed before yolo-world , but the interesting point of how we deal with the trained data for every model , for example : test on owl-vit or clip or yolo-world in detection task for complex detection task , then compare them with a trained yolo on this task, how these model would perform in words of accuarcy and latency ,also we dont talk abould complex envs , like rotation , homography , etc... , you will see the difference. the yolo wins , but the idea from this survey was how about tradeoff between embeddings and training can we use embeddings and not train at all, for image-language learning some of lack of percision and recall cause of training , but can we can fill this gap using image-image learing (image embbedings) 
- for owl-vit it's better than clip cause how the vit was trained , but a new problem to think about can we use embbedings in detecting objects in alternate of training models for object detection task like YOLO


## TODO 

- [x] Task 1: Survey between clip and owl-vit.
- [ ] Task 2: survey between clip and owl-vit and yolo-world and trained yolo.
- [ ] Task 3: publishing results.

## Features

- Applying Zero-shot object detection using Clip , Owl-VIT

## Installation

To use ZSOD, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Alaaeldinn/ZSOD.git
   ```

2. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

## Uses 

- for example : apply detection anomaly objects in factories etc , when detecting make the model to write description !!! 
