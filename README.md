# RobotGPT
 Use llava to add other multi modal ability
# The current progress 
## Image
Image tower pretrain code is tested.   
The dataset is CC3M,  the same as the LLaVA ones.  
Dataset web: [LLaVA pretrain Images Json](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K)  
Images:    
```
链接：https://pan.baidu.com/s/1eQbfEK07DaKo2HzdFXwN7A 
提取码：pu68 
--来自百度网盘超级会员V7的分享
```
## Audio
Audio tower pretrain code tested.  
(May have some problem in sampling rate calculation for the audio file reader, need further check)    
The audio dataset is audiocaps.  
Audio Dataset:   
```
链接：https://pan.baidu.com/s/1TQgvo056Vejj-Y95dqoZpQ 
提取码：88xc 
--来自百度网盘超级会员V7的分享
```
Detail setting for Json file: /pretrain/audiocap_setup.zip  
In check_files.py, it would be suggested to set files at least larger than 100k.  
That can help reducing dataset problem.
## Cli inference
Not finished testing yet
