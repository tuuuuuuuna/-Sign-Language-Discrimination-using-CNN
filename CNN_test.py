from PIL import Image
import os,glob,numpy as np
from keras.models import load_model

categories=[chr(ord("A")+i) for i in range(0,26)]
nb_classes = len(categories)
caltech_dir = "./test"
image_w = 50
image_h = 50

pixels = image_h * image_w * 3

x = []
filenames = []
files = glob.glob(caltech_dir+"/*.jpg")

for idx,alpha in enumerate(categories):
    image_dir = caltech_dir + "/" + alpha
    files = glob.glob(image_dir+"/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)

        x.append(data)
        filenames.append(f)

x = np.array(x)
model = load_model('./model/multi_img_classification.model')

prediction = model.predict(x)
np.set_printoptions(formatter = {'float': lambda x : "{0:0.3f}".format(x)})
cnt = 0
print(float(ord(categories[1])- 65))
for i in prediction:
    pre_ans = i.argmax()
    pre_ans_str = ''

    for j in categories:
        if pre_ans == float((ord(j) - 65)):
            pre_ans_str = j
            break
    print("해당" + filenames[cnt]+"이미지는 "
          +pre_ans_str + "으로 추정됩니다.")
    cnt += 1

