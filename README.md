# NARUTO_game
摄像头识别手势来释放忍术

##  需要环境：  
* tensorflow1.1
* keras
* opencv
* python3
* ffmpeg
* PIL
* pathlib
* shutil
* imageio
* numpy
* json
* pygame

## 使用方法
用jupyternotebook打开tutorial.ipynb文件，按照里面的提示，一步一步运行

## 注意事项
* 测试机器的摄像头每秒约30帧，代码中的检测速率也按这个帧数设置的，某些笔记本的摄像头fps没有那么高，可能导致识别速率慢
* 测试时我用的vscode把当前工作路径设置为 NARUTO_game 这个主文件夹，并以此设置相关的相对路径，若直接cd到model文件夹来运行predict.py文件，需要手动调整源码中的相对路径
