import tensorflow.compat.v1 as tf # use import tensorflow as tf incase you use TF1
import tensornets as nets
import cv2
import numpy as np
from PIL import Image, ImageDraw

tf.disable_v2_behavior() # incase you use TF2, if TF1 then mark this line

MODEL_TRESHOLD = .73
MODEL_CAR_CLASS = 2 # car class in tensornets
MODEL_WIDTH = 416
MODEL_HEIGHT = 416

def find_cars_in_image(image_name):
    
    inputs = tf.placeholder(tf.float32, [None, MODEL_WIDTH, MODEL_HEIGHT, 3]) 
    model = nets.YOLOv3COCO(inputs, nets.Darknet19, reuse = tf.AUTO_REUSE)
    
    with tf.Session() as sess:
    
        with Image.open(image_name) as img:
            
            original_width, original_height = img.size

            sess.run(model.pretrained())

            img_resized = img.resize((MODEL_WIDTH, MODEL_HEIGHT))
            img_reshaped = np.array(img_resized).reshape(-1,MODEL_WIDTH, MODEL_HEIGHT, 3)
            preds = sess.run(model.preds, {inputs: model.preprocess(img_reshaped)})
            boxes = model.get_boxes(preds, img_reshaped.shape[1:3])
            boxes1 = np.array(boxes)

            if len(boxes1) != 0:
                
                json_data = ''
                object_numer = 1
                
                for i in range(len(boxes1[MODEL_CAR_CLASS])):
                    box = boxes1[MODEL_CAR_CLASS][i]
                    if boxes1[MODEL_CAR_CLASS][i][4] >= MODEL_TRESHOLD:
                        
                        original_size_x0 = (original_width / MODEL_WIDTH) * box[0]
                        original_size_y0 = (original_height / MODEL_HEIGHT) * box[1]
                        original_size_x1 = (original_width / MODEL_WIDTH) * box[2]
                        original_size_y1 = (original_height / MODEL_HEIGHT) * box[3]
                        shape = [(original_size_x0, original_size_y0), (original_size_x1, original_size_y1)]
                        
                        if json_data != '':
                            json_data = json_data + ','
                        
                        json_data = json_data + '\n\t"object ' + str(object_numer) + '" : "' + str(original_size_x0) + ',' + str(original_size_y0)
                        json_data = json_data  + ',' + str(original_size_x1) + ',' + str(original_size_y1) + '"'
                        
                        object_numer = object_numer + 1
                        img_with_rectangle = ImageDraw.Draw(img)
                        img_with_rectangle.rectangle(shape, outline = "red")
                
                json_data = '{' + json_data + '\n}'
                
                with open('boxes.json', 'w') as f:
                    f.write(json_data)
                img.save(image_name, format=None)

def find_cars_in_video(video_name):

    inputs = tf.placeholder(tf.float32, [None, MODEL_WIDTH, MODEL_HEIGHT, 3]) 
    model = nets.YOLOv3COCO(inputs, nets.Darknet19, reuse=tf.AUTO_REUSE)
    
    with tf.Session() as sess:
    
        sess.run(model.pretrained())
        
        cap = cv2.VideoCapture(video_name)
        original_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        img_array = []
        frame_num = 1
        
        while(frame_num < total_frames):

            ret, frame = cap.read()
            img_resized = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
            img_reshaped = np.array(img_resized).reshape(-1, MODEL_WIDTH, MODEL_HEIGHT, 3)
            preds = sess.run(model.preds, {inputs: model.preprocess(img_reshaped)})
            boxes = model.get_boxes(preds, img_reshaped.shape[1:3])
            
            boxes1 = np.array(boxes)
            
            if len(boxes1) != 0:
                for i in range(len(boxes1[MODEL_CAR_CLASS])):
                    box = boxes1[MODEL_CAR_CLASS][i]
                    if boxes1[MODEL_CAR_CLASS][i][4] >= MODEL_TRESHOLD:
                        
                        original_size_x0 = int((original_width / MODEL_WIDTH) * box[0])
                        original_size_y0 = int((original_height / MODEL_HEIGHT) * box[1])
                        original_size_x1 = int((original_width / MODEL_WIDTH) * box[2])
                        original_size_y1 = int((original_height / MODEL_HEIGHT) * box[3])
                        cv2.rectangle(frame, (original_size_x0, original_size_y0), (original_size_x1, original_size_y1), (0, 255, 0), 3)
                        
            img_array.append(frame)
            frame_num = frame_num + 1
        
        new_video_name = video_name[0:video_name.rfind('.')] + '_With_Boxes.avi'
        
        video = cv2.VideoWriter(new_video_name, cv2.VideoWriter_fourcc(*'DIVX'), 15, (original_width, original_height))

        for i in range(len(img_array)):
            video.write(cv2.resize(img_array[i], (original_width, original_height)))
        video.release()




