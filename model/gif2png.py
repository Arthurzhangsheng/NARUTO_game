#-*- coding: UTF-8 -*-    
import os  
from PIL import Image  
  
def analyseImage(path):  
    ''''' 
    Pre-process pass over the image to determine the mode (full or additive). 
    Necessary as assessing single frames isn't reliable. Need to know the mode  
    before processing all frames. 
    '''  
    im = Image.open(path)  
    results = {  
        'size': im.size,  
        'mode': 'full',  
    }  
    try:  
        while True:  
            if im.tile:  
                tile = im.tile[0]  
                update_region = tile[1]  
                update_region_dimensions = update_region[2:]  
                if update_region_dimensions != im.size:  
                    results['mode'] = 'partial'  
                    break  
            im.seek(im.tell() + 1)  
    except EOFError:  
        pass  
    return results  
  
  
def processImage(path):  
    ''''' 
    传入GIF图的完整路径名字,自动在该位置创建png文件夹来保存所有图片
    
    '''  
    mode = analyseImage(path)['mode']  
      
    im = Image.open(path)  
  
    i = 0  
    p = im.getpalette()  
    last_frame = im.convert('RGBA')  
      
    try:  
        while True:  
            print ("saving %s (%s) frame %d, %s %s" % (path, mode, i, im.size, im.tile)  )
              
            ''''' 
            If the GIF uses local colour tables, each frame will have its own palette. 
            If not, we need to apply the global palette to the new frame. 
            '''  
            if not im.getpalette():  
                im.putpalette(p)  
              
            new_frame = Image.new('RGBA', im.size)  
              
            ''''' 
            Is this file a "partial"-mode GIF where frames update a region of a different size to the entire image? 
            If so, we need to construct the new frame by pasting it on top of the preceding frames. 
            '''  
            if mode == 'partial':  
                new_frame.paste(last_frame)  
              
            new_frame.paste(im, (0,0), im.convert('RGBA'))  
            fp = "/".join(path.split('/')[:-1]) + '/png'
            make_path(fp)
            fp = "/".join(path.split('/')[:-1]) + '/png/' + 'action'
            new_frame.save('%s-%02d.png' % (fp , i), 'PNG')  
  
            i += 1  
            last_frame = new_frame  
            im.seek(im.tell() + 1)  
    except EOFError:  
        pass  
def make_path(p):  
    if not os.path.exists(p):       # 判断文件夹是否存在  
        os.makedirs(p)          # 创建文件夹
               
  
  
def main():  
    processImage('image/shuidun/shuidun.gif')  
      
  
if __name__ == "__main__":  
    main() 