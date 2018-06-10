# coding=utf-8
from keras import models
import numpy as np
import cv2 
import json
import os
from PIL import Image, ImageDraw, ImageFont  
import pygame,time

def load_config(fp):
    with open(fp,encoding='UTF-8') as f:
        config = json.load(f, encoding='UTF-8')
        indices = config['indices']
        input_size = config['input_size']
        return indices, input_size


def decode(preds, indices):
    results = []
    for pred in preds:
        index = pred.argmax()
        result = indices[str(index)]
        results.append(result)
        result = results[0]
    
    return result


def preprocess(arr, input_size):
    input_size = tuple(input_size)
    # resize
    x = cv2.resize(arr, input_size)
    # BGR 2 RGB
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = np.expand_dims(x, 0).astype('float32')
    x /= 255
    return x

def put_text_on_img(img,
                    text='文字信息',
                    font_size = 50,
                    start_location = (100, 0),
                    font_color = (255, 255, 255),
                    fontfile = 'model/font.ttf'):
    '''
    读取opencv的图片，并把中文字放到图片上
       
    font_size = 100             #字体大小
    start_location = (0, 0)     #字体起始位置
    font_color = (0, 0, 0)      #字体颜色
    fontfile = 'model/font.ttf' #字体文件
    '''
    # cv2读取图片  
    
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2和PIL中颜色的hex码的储存顺序不同  
    pilimg = Image.fromarray(cv2img)  
    
    # PIL图片上打印汉字  
    draw = ImageDraw.Draw(pilimg) # 图片上打印  
    font = ImageFont.truetype(fontfile, font_size, encoding="utf-8") # 参数1：字体文件路径，参数2：字体大小  
    draw.text(start_location, text, font_color, font=font) # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体  
    
    # PIL图片转cv2 图片  
    convert_img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)  
    # cv2.imshow("图片", cv2charimg) # 汉字窗口标题显示乱码  
    return convert_img

def playBGM():
    bgm_path = r'audio/BGM.mp3'
    pygame.mixer.init()


    pygame.mixer.music.load(bgm_path)
    pygame.mixer.music.set_volume(0.2) 
    pygame.mixer.music.play(loops=-1)

def playsound(action):
    sound_path1 = 'audio/test1.wav'
    sound_path2 = 'audio/test2.wav'
    sound_path3 = 'audio/huituzhuansheng.wav'
    sound_path4 = 'audio/yingfenshen.wav'
    
    if action == "寅":
        sound1 = pygame.mixer.Sound(sound_path2)
        sound1.set_volume(0.3)
        sound1.play()
        
    elif action == "申":
        sound1 = pygame.mixer.Sound(sound_path1)
        sound1.set_volume(0.5)
        sound1.play()
    elif action == '酉':
        sound1 = pygame.mixer.Sound(sound_path3)
        sound1.set_volume(1)
        sound1.play()        
    elif action == "丑":
        sound1 = pygame.mixer.Sound(sound_path4)
        sound1.set_volume(1)
        sound1.play()  
        
    else:
        pass    

def add_gif2cap(cap, pngimg):
    # I want to put logo on top-left corner, So I create a ROI
    rows1,cols1,channels1 = cap.shape
    rows,cols,channels = pngimg.shape

    roi = cap[(rows1-rows)//2:(rows1-rows)//2+rows, (cols1-cols)//2:(cols1-cols)//2+cols ]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(pngimg,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(pngimg,pngimg,mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    cap[(rows1-rows)//2:(rows1-rows)//2+rows, (cols1-cols)//2:(cols1-cols)//2+cols] = dst

    return cap

def add_gif2cap_with_action(action, cap, png_num):
    if action == "寅":
        pngpath = 'image/shuilongdan/png/action-%02d.png'%(png_num)
        pngimg = cv2.imread(pngpath)
        pngimg = cv2.resize(pngimg,None,fx=0.8, fy=0.8, interpolation = cv2.INTER_CUBIC)
        cap = add_gif2cap(cap, pngimg)
        return cap
    else:
        return cap



def main():
    indices, input_size = load_config('model/config.json')
    model = models.load_model('model/NARUTO.h5')
    cap = cv2.VideoCapture(0)
    counter = 0      
    counter_temp = 0 #计数器
    action = "子"
    playBGM()
    png_num = 1 #用于计数动画图片序号的变量
    while True:
        _, frame_img = cap.read()
        # predict
        x = preprocess(frame_img,input_size)
        y = model.predict(x)
        action = decode(y,indices)
  
        #播放音效,且每次播放间隔50个帧
        counter+=1
        if counter == 2:
            #触发音效
            playsound(action)             
            counter += 1
        if counter == 50:
            counter = 0
       
        #显示动作名  
        frame_img = put_text_on_img(
            img= frame_img,
            text= "當前動作:"+action,
            font_size = 50,
            start_location = (0, 100),
            font_color = (255, 150, 0)

        )
        #触发动画
        if action == "寅":
            frame_img = add_gif2cap_with_action(action, frame_img, png_num)
            png_num += 1
            if png_num >=37:#水龙弹动画有37帧
                png_num=0 

        #show image
        cv2.imshow('webcam', frame_img)
        
        #按Q关闭窗口
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
   # playBGM()