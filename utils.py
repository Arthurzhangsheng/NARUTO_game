from pathlib import Path
import shutil
import imageio
from PIL import Image
import numpy as np
import math
import json
import os


def process_videos(fps, target_size):
    '''把data/video文件夹内的视频以fps采样率转换成图片，
    并且按照60%、20%、20%的比例随机拆分存入train、val、test文件夹中。
    video文件夹中的视频需要每一类新建一个单独文件夹进行保存。
    fps: int
    target_size: tuple(width, height)
    '''

    # 获取物体种类目录
    base_dir = Path('./data')
    print(base_dir)
    video_dir = base_dir / 'video'
    print(video_dir)
    categories = get_child_dir_names(video_dir)#获取video文件夹下的所有子文件夹名字，并存入categories列表
    print('共找到{}种类别: {}'.format(len(categories), categories))

    # 新建data/image及其种类子目录
    image_dir = base_dir / 'image'#指定image文件夹的位置
    makedir(image_dir)#创建image文件夹
    for ctg in categories:#读取categories中保存的所有子文件夹名字
        path = image_dir / ctg #在image文件夹中创建这些子文件夹的路径
        makedir(path)#创建这些子文件夹

    # 把video中的视频转成jpg图片存到image中
    # 遍历data/video下的每一个类别目录
    for ctg in categories:
        videos = [
            path for path in (video_dir / ctg).iterdir() if path.is_file()
        ]#创建一个列表，保存某个文件夹里的所有视频名字
        for video in videos:
            try:
                print('正在处理{}'.format(video))
                # 从视频中提取图片
                imgs = img_generator(video, fps)#从指定的视频文件中，按指定的fps来读取图片的路径
                for i, img in enumerate(imgs):#依次获取imgs的图片路径名字来读取图片
                    # resize to target_size
                    
                    img = img.resize(target_size)#重新定图片尺寸
                    img_name = '{}_{}.jpg'.format(video.stem, i)#生成图片的名字
                    img_path = image_dir / ctg / img_name#生成图片的保存位置 data/image/xu/xu_35.jpg
                    img.save(img_path)#保存图片
            except:
                print("处理{}时遇到神秘bug，跳过此文件".format(video))
                continue
    # 创建train, val, test及其子目录
    train_dir = base_dir / 'train'#先生成文件夹的位置信息
    val_dir = base_dir / 'val'
    test_dir = base_dir / 'test'
    for d in [train_dir, val_dir, test_dir]:
        makedir(d)#创建文件夹
        for ctg in categories:
            makedir(d / ctg)#创建子文件夹，类似train/xu

    # 随机拆分
    # 遍历data/image下的每个类别目录
    for ctg in categories:
        cur_dir = image_dir / ctg#构建子文件夹的完整路径
        jpgs = np.array([f for f in cur_dir.iterdir() if f.match('*.jpg')])#构建包含所有图片路径名称的列表
        train, val, test = random_split(jpgs)#把这个列表先打乱顺序，再切割成三段返回
        move(train, train_dir / ctg)#移动图片到指定位置
        move(val, val_dir / ctg)
        move(test, test_dir / ctg)

    # remove image_dir
    shutil.rmtree(image_dir)#删除了image文件夹

    print('处理完成')


def get_child_dir_names(path):
    return [d.name for d in path.iterdir() if d.is_dir()]
#zheshizaiganma

def count_jpgs(path, recursive=True):
    jpg_generator = path.rglob('*.jpg')
    return sum(1 for _ in jpg_generator)


def move(paths, dst_dir):
    '''把paths中的文件移动到dst目录
    Parameters
    paths: ndarray(1,) 每个元素是一个Path object
    dst_dir: 目标文件夹
    '''
    for path in paths:
        dst = dst_dir / path.name
        shutil.move(path, dst)
        # print('{}已移动到{}'.format(path, dst))


def random_split(x):
    '''把x随机打乱顺序，然后按照60%,20%,20%的比例切割成三份
    Parameters
    x: numpy.ndarray (1,)
    Returns
    train: numpy.ndarray (1,)
    val: numpy.ndarray (1,)
    test: numpy.ndarray (1,)
    '''
    np.random.shuffle(x)
    n = len(x)
    train_end = math.ceil(n * 0.6)
    val_end = math.ceil(n * 0.2) + train_end
    train = x[:train_end]
    val = x[train_end:val_end]
    test = x[val_end:]
    return train, val, test


def makedir(path):
    if not path.exists():
        path.mkdir()
    #     print('{} 创建成功'.format(path))
    # else:
    #     print('{} 已存在'.format(path))


def img_generator(videopath, fps):
    '''把视频转成图片
    Parameters
    videopath: Path object
    fps: int
    Return
    generator 每次返回一张图片（PIL.Image object）
    '''
    assert videopath.is_file()
    reader = imageio.get_reader(videopath._str)
    meta = reader.get_meta_data()
    sampling_rate = meta['fps'] / fps
    indices = filter(lambda i: 0 <= i % sampling_rate < 1,
                     range(meta['nframes']))#过滤出所有需要被读取的帧
    for index in indices:
        img = reader.get_data(index)#读取具体的某一帧的图
        # convert to PIL Image
        yield Image.fromarray(img)


def preprocess(img, target_size):
    '''缩放、转成ndarray、除以255
    Arguments
    img - PIL Image object
    target_size - tuple(width, height)
    Return
    x - ndarray(height, width, channel, 1)
    '''
    img = img.resize(target_size, Image.ANTIALIAS)
    # 转成ndarray，并且增加一个维度
    x = np.expand_dims(img, 0).astype('float')
    x /= 255
    return x

def show_pred(y, class_indices_reverse):
    '''
    Arguments
    y - ndarray(1, n)
    class_indices_reverse - 以编号为索引的类别名称字典
    '''
    for i in range(len(y)):
        print('这张图片有 {:.2f}% 的可能性是 {}'.format(y[i]*100, class_indices_reverse[i]))
    
    

def save_confg(indicies, input_size, fp):
    '''把字典存成.json文件
    Arguments
    indicies - dict
    fp - file path
    '''
    config = {
        'indices': indicies,
        'input_size': input_size
    }
    with open(fp, 'w', encoding='UTF-8') as f:
        json.dump(config, f,ensure_ascii=False)