# coding=utf-8
from keras import models
import numpy as np
import cv2 
import json
import os
from PIL import Image, ImageDraw, ImageFont  


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
    return results


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
                    font_size = 100,
                    start_location = (0, 0),
                    font_color = (0, 0, 0),
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

def main():
    indices, input_size = load_config('model/config.json')
    print(indices)
    print(input_size)
    model = models.load_model('model/NARUTO.h5')
    cap = cv2.VideoCapture(0)
    while True:
        s, frame_img = cap.read()
        # predict
        x = preprocess(frame_img,input_size)
        y = model.predict(x)
        action_name = decode(y,indices)
        print(action_name)
        # 实时显示当前动作名
        frame_img = put_text_on_img(img=frame_img,
                            text=action_name[0],
                            font_size=50,
                            font_color=(255,255,255),
                            start_location=(100,0))

        #show image
        cv2.imshow('webcam', frame_img)
        #按Q关闭窗口
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()