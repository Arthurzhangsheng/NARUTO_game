from keras import models
import numpy as np
import cv2 as cv
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def load_config(fp):
    with open(fp) as f:
        config = json.load(f)
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
    x = cv.resize(arr, input_size)
    # BGR 2 RGB
    x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
    x = np.expand_dims(x, 0).astype('float32')
    x /= 255
    return x


def main():
    indices, input_size = load_config('model/config.json')
    model = models.load_model('model/NARUTO.h5')
    cap = cv.VideoCapture(0)
    while True:
        s, f = cap.read()
        # predict
        x = preprocess(f,input_size)
        y = model.predict(x)
        r = decode(y)
        # plot result
        cv.putText(
            img=f,
            text=r[0],
            org=(250, 50),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 98, 255),
            thickness=2)
        # show image
        cv.imshow('webcam', f)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()