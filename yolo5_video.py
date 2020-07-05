# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:31:23 2020

@author: Harinath
"""

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import imageio
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils_video import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
#from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
import cv2
from PIL import Image, ImageDraw, ImageFont
%matplotlib inline





def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .3):
    box_scores = box_confidence*box_class_probs
    box_classes = K.argmax(box_scores,axis=-1)
    box_class_scores = K.max(box_scores,axis=-1)
    filtering_mask = box_class_scores>=threshold
    scores = tf.boolean_mask(box_class_scores,filtering_mask)
    boxes = tf.boolean_mask(boxes,filtering_mask)
    classes = tf.boolean_mask(box_classes,filtering_mask)  
    return scores, boxes, classes





def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):    
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes,iou_threshold=0.5,name=None)
    scores = K.gather(scores,nms_indices)
    boxes = K.gather(boxes,nms_indices)
    classes = K.gather(classes,nms_indices)    
    return scores, boxes, classes



def yolo_eval(yolo_outputs, image_shape , max_boxes=20, score_threshold=.3, iou_threshold=.5):
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .3)
    boxes = scale_boxes(boxes, image_shape)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = 20, iou_threshold = 0.5)
    return scores, boxes, classes



def yolo_head(feats, anchors, num_classes):
    num_anchors = len(anchors)
    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])
    conv_dims = K.shape(feats)[1:3]  # assuming channels last.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])
    conv_width_index = K.tile(K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))    
    feats = K.reshape(feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_xy = K.sigmoid(feats[..., :2])
    box_wh = K.exp(feats[..., 2:4])
    box_class_probs = K.softmax(feats[..., 5:])
    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims

    return box_confidence, box_xy, box_wh, box_class_probs

def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return K.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])



def predict(sess, image_file):
    # Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    ### START CODE HERE ### (≈ 1 line)
    out_scores, out_boxes, out_classes = sess.run(fetches=[scores,boxes,classes],
       feed_dict={yolo_model.input: image_data,
                  K.learning_phase():0
    })
    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    output_image = imageio.imread(os.path.join("out", image_file))
    imshow(output_image)
    
    return out_scores, out_boxes, out_classes


def process_image(sess,image,text_file):
    model_image_size=(608,608)
    resized_image=image.resize(tuple(reversed(model_image_size)),Image.BICUBIC)
    image_data=np.array(resized_image,dtype='float32')
    image_data/=255.
    image_data=np.expand_dims(image_data,0)
    out_scores, out_boxes, out_classes = sess.run(fetches=[scores,boxes,classes],
       feed_dict={yolo_model.input: image_data,
                  K.learning_phase():0
    })
    print('Found {} boxes'.format(len(out_boxes)))
    print('Found {} boxes'.format(len(out_boxes)),file=text_file)
    colors=generate_colors(class_names)
    draw_boxes(image,out_scores,out_boxes,out_classes,class_names,colors,text_file)
    return image
    
def predict_video(sess,video_file):
    video_path=os.path.join('videos',video_file)
    cap=cv2.VideoCapture(video_path)
    text_file=open("out.txt",'w+')
    sz=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc=cv2.VideoWriter_fourcc(*'mpeg')
    out=cv2.VideoWriter()
    out.open(os.path.join("out",video_file),fourcc,20,sz,True)
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        frame= Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        image= process_image(sess,frame,text_file)
        image=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        cv2.imshow("detection",image)
        out.write(image)
        if cv2.waitKey(1)& 0xff == ord('q'):
            break
        
    out.release()
    cap.release()
    text_file.close()


sess = K.get_session()
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (1080.,1920.) 

yolo_model = load_model("model_data/yolov2.h5")


yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
#out_scores, out_boxes, out_classes = predict(sess, "images14.jpeg")
video_file="traffic.mp4"
predict_video(sess,video_file)


