# Non Local Network implementation on the UCF-101 dataset
This repository is a modification of the Two-Stream network based on ![Jeffrey Huang's work](https://github.com/jeffreyhuang1/two-stream-action-recognition  
):  
Also utilizing ![AlexHex7's PyTorch implementation](https://github.com/AlexHex7/Non-local_pytorch/blob/master/Non-Local_pytorch_0.3.1/lib/backup/non_local_simple_version.py) of the Non-Local Block to enhance the spatial CNN of the Two-Stream network:  

The main added feature of this repository is adding an inference method to the networks so you can see the model's predictions
(Top-5 and their score) in real-time on a webcam feed

### Demo
[![Link to demo video](https://raw.githubusercontent.com/danbochman/Real-Time-Action-Recognition/master/demo/screenshot.png)](https://youtu.be/21HJVh29pY8) <br>
You can click the image to view the demo video ^

## Usage

### Prerequisites
Please note that this is repository was built on **Python 2.7**.
Unfortunately, at the time of creating this repo, I did not have the best Git protocols and haven't made a proper requirements.txt - My apologies.

### Training
If you want to train the model from scratch you need to download the UCF-101 data, I recommend visiting Jefferey's Huang repository linked above and follow his detailed instructions.

### Inference
If you just want to run inference download the pre-trained model here:  
[Link to ResNet101 trained on UCF-101](https://drive.google.com/drive/folders/1gVB5StqgoDJ3IxHUn7zoTzTNxzz3du3d?usp=sharing)

Then run
```
python spatial_cnn_gpu --resume /PATH/TO/model_best.pth.tar --demo
```
You can run a cpu only version just by changing the script's name to spatial_cnn_cpu.py
The best real-time results come from running only the Spatial CNN without the Temporal Stream on a GPU

### Pre-trained Weights
We didn't include the pre-trained weights to the Non-Local-Network version because we didn't observe any improvement in performance by adding the Non-Local Blocks (NLBs).
We believe that very big batch-sizes are required for NLBs to contribute to the precision's score, which we didn't have the resources for.

## Reference Papers
*  [[1] Two-stream convolutional networks for action recognition in videos](http://papers.nips.cc/paper/5353-two-stream-convolutional)
*  [[2] Non-local Neural Networks](https://arxiv.org/abs/1711.07971)
