# coding=utf-8
from wxpy import *
import cv2
import os




def face(name):
    print('processing')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    print('file found')
    count = 0
    print(name)
    img = cv2.imread(name)
    #print(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(faces)
    for (x, y, w, h) in faces:
        count += 1
        print(count)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)
        print('11111111111111')
        font = cv2.FONT_HERSHEY_SIMPLEX
        print('22222222222222222')
        print(x,y,w,h)
        print(gray)
        #roi_gray = gray[y:y + h / 2, x:x + w]
        #print(roi_gray)
        print('333333333333333333')
        #roi_color = img[y:y + h / 2, x:x + w]
        #print(roi_color)
        print('22222222222222')

    print('face count done')
    cv2.imwrite("face_detected_1.jpg", img)  # 保存已经生成好的图片
    print('photo gen')
    print(count)
    return count  # 返回人脸总数


bot = Bot(cache_path=True)

@bot.register(Friend, PICTURE)
def face_msg(msg):
    image_name = msg.file_name
    friend = msg.chat
    print( msg.chat)
    print( '接收图片')
    # face(image_name)
    msg.get_file('' + msg.file_name)
    print('got file')
    count = face(image_name)
    if count == 0:
        msg.reply(u'未检测到人脸')
    else:
        msg.reply_image("face_detected_1.jpg")
        msg.reply(u"检测到%d张人脸" % count)
    os.remove(image_name)
    os.remove("face_detected_1.jpg")


embed()
