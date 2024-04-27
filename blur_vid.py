
"""
**Aim:** This is the final code of video blur along with UI

**Author:** Shalu Singh

**Starting Date:** 12/9/23

**Ending Date:** 14/1/24
"""

# import libraries
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import pandas as pd
import keras
import gradio
from concurrent.futures import ThreadPoolExecutor
from moviepy.editor import VideoFileClip, concatenate_videoclips

# path to ouput video
out_video_path = 'blured_op_video.mp4'

# class label
coco_classes = {
0: 'unlabeled',
1: 'person',
2: 'bicycle',
3: 'car',
4: 'motorcycle',
5: "airplane",
6: "bus",
7: "train",
8: "truck",
9: "boat",
10:" traffic light",
11: "fire hydrant",
12: "street sign",
13: "stop |sign",
14: "parking meter",
15: "bench",
16: "bird",
17: "cat",
18: "dog",
19: "horse",
20: "sheep",
21: "cow",
22: "elephant",
23:" bear",
24: "zebra",
25: "giraffe",
26: "hat",
27: "backpack",
28: "umbrella",
29: "shoe",
30: "eye glasses",
31: "handbag",
32:" tie",
33: "suitcase",
34:" frisbee",
35: "skis",
36: "snowboard",
37: "sports ball",
38: "kite",
39: "baseball bat",
40: "baseball glove",
41: "skateboard",
42: "surfboard",
43: "tennis racket",
44: "bottle",
45: "plate",
46: "wine glass",
47: "cup",
48: "fork",
49: "knife",
50: "spoon",
51: "bowl",
52: "banana",
53:"apple",
54:"sandwich",
55:" orange",
56: "broccoli",
57: "carrot",
58: "hot dog",
59:' pizza',
60: "donut",
61: 'cake',
62: "chair",
63: "couch",
64: "potted plant",
65: "bed",
66: "mirror",
67: "dining table",
68: "window",
69: "desk",
70: "toilet",
71: "door",
72: "tv",
73:" laptop",
74: "mouse",
75: "remote",
76:" keyboard",
77: "cell phone",
78: "microwave",
79: "oven",
80: "toaster",
81: "sink",
82: "refrigerator",
83: "blender",
84: "book",
85:"clock",
86: "vase",
87: "scissors",
88: "teddy bear",
89: "hair drier",
90: "toothbrush",
}

coco_encode = {value:key for key,value in coco_classes.items()}
coco_labels = list(coco_classes.values())

# function: blur the image
def blur_image(image = None,coordinates = None,blur_value = 3):
  #print('*********** INSIDE [blur_image()] *********]')
  img = image.copy() # copy the image to work on new image
  if (coordinates is not None):
    #print('Performing image blur operation...')
    for coord in (coordinates):
      ymin,xmin,ymax,xmax = coord
      #print('Image shape:',img.shape)
      # Extract region of intrest
      Y_min,X_min,Y_max,X_max = int(ymin*img.shape[0]),int(xmin*img.shape[1]),int(ymax*img.shape[0]),int(xmax*img.shape[1])
      #print('Y_min,Y_max',Y_min,Y_max)
      #print('X_min,X_max',X_min,X_max)
      roi = img[Y_min:Y_max,X_min:X_max]
      #show_img(roi,'Original_roi')
      # blur the extracted img using Gausian blur
      try:
        roi = cv2.GaussianBlur(roi,ksize = (blur_value,blur_value),sigmaX = 0)
        #show_img(roi,title='blured roi')
        # replace the original roi with blured_roi
        img[Y_min:Y_max, X_min:X_max] = roi
      except:
        pass

    return img

# function: filter detection boxs
def filter_detection(detector_output,select_classes,thr = 0.6):
 # print('********* INSIDE [filter_detection()] **********')
  detection_boxs = detector_output['detection_boxes']
  detection_class = detector_output['detection_classes']
  detection_scores = detector_output['detection_scores']
 # get the masking to select classes which user choosed
  masked_classes = np.isin(detection_class,select_classes)

  # select only selected classes
  detection_class = detection_class[masked_classes]
  detection_boxs = detection_boxs[masked_classes]
  detection_scores = detection_scores[masked_classes]

  # filter the detection boxses based on threshold
  selected_scores = detection_scores[detection_scores >= thr]
  selected_class = detection_class[detection_scores >= thr]
  selected_boxs = detection_boxs[detection_scores >= thr].numpy()

  return selected_boxs,selected_class,selected_scores

# get the input video
# load video from local disk
def load_input(ip_path):
  #print('******* INSIDE [load_input] ********')
  try:
    cap = cv2.VideoCapture(ip_path)
    print('Video loaded successfully!')
    return cap
  except:
    print("Failed! to load video")

#function: get video property like frame_width,frame_heigh,frame_per_second(fps),codecc
def out_video(cap):
  #print('******** INSIDE [out_video] ***********')
  frame_width = int(cap.get(3)) # width of the fames in the video
  frame_height = int(cap.get(4)) # height of the frame in the video
  fps = int(cap.get(5)) # frame per second
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  video_duration =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT))/ fps
  codecc = cv2.VideoWriter_fourcc(*'mp4V') # codecc for output video ( h264  codecc)
  # video property info
  print('Frame Width:',frame_width)
  print('Frame height:',frame_height)
  print('Frame Per Second:',fps)
  print('Total frames:',total_frames)
  print('video_duration: {} minutes'.format(round(video_duration/60),2))
  # VideoWriter object to save blured video
  out = cv2.VideoWriter(out_video_path,codecc,fps,(frame_width,frame_height))
  return out,fps,total_frames,video_duration

# function: to get time range to perfrom blur
def time_range(start_time,end_time):
  #print('*********** INSIDE [time_range()] ************')
  start_time,end_time = start_time,end_time # change to second(s) format
  return start_time,end_time

# function: to check if time range is valid or not
def is_valid_time_range(start_time,end_time,video_duration):
  #print('********** INSIDE [valid_time_range()] *************')
  return (0 <= start_time < end_time <= video_duration)


# load model

object_detection_model = tf.saved_model.load("Object_detection_model.h5")



def blur_video(input_video_path, u_classes, start_time, end_time):
    print('STARTING OF PROCESSING...')
    print("u_classes:",u_classes,type(u_classes))
    label_encode = np.array([coco_encode[i] for i in u_classes], dtype='float16')
    print('label_encode:',label_encode,type(label_encode))
    cap = load_input(ip_path=input_video_path)
    out, fps, total_frames, video_duration = out_video(cap)
    start_time, end_time = time_range(start_time, end_time)

    if is_valid_time_range(start_time, end_time, video_duration):
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        print('Start Frame:', start_frame)
        print('End Frame:', end_frame)

        with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers as needed
            futures = []
            for i in range(total_frames):
                ret, frame = cap.read()
                if ret:
                    frame = tf.expand_dims(frame, axis=0)
                else:
                    break

                if start_frame <= i <= end_frame:
                    print('Blured_frame:',i)
                    future = executor.submit(blur_process, frame, label_encode)
                    futures.append(future)
                else:
                    out.write(frame[0].numpy())

            for future in futures:
                blured_img = future.result()
                out.write(blured_img)

        cap.release()
        out.release()

    return out_video_path

def blur_process(frame,l_encoder,blur_value):
    print('label_encode',l_encoder)
    frame = np.expand_dims(frame,axis = 0)
    detector_output = object_detection_model(frame)
    boxes,classes,scores = filter_detection(detector_output,l_encoder)
    blured_img = blur_image(frame[0],boxes,blur_value)
    return blured_img



def process_and_concat_video(input_video_path,u_classes,blur_value,start_time, end_time):
    label_encode = np.array([coco_encode[i] for i in u_classes],dtype = 'float16')
    # Load the full video clip
    full_video_clip = VideoFileClip(input_video_path)

    # Process the specified part of the video
    processed_clip = full_video_clip.subclip(start_time, end_time).set_duration(end_time - start_time)
    processed_clip = processed_clip.fl_image(lambda frame: blur_process(frame,label_encode,blur_value))
    print('final clip fps:',full_video_clip.fps)
    print('processed_clip fps:',processed_clip.fps)

    # Concatenate the processed and unprocessed parts
    final_clip = concatenate_videoclips([full_video_clip.subclip(0, start_time),
                                         processed_clip,
                                         full_video_clip.subclip(end_time, None)])

    final_clip.set_fps = 25  # Assuming desired FPS is 25


    # Write the final video to an output file with the specified fps
    out_video_path = "output_video.mp4"
    final_clip.write_videofile(out_video_path, codec="h264", audio_codec="aac",fps = 25)

    return out_video_path


if  __name__ == "__main__":
  import gradio as gr
  iface = gr.Interface(
      fn=process_and_concat_video,
      inputs=[
          gr.Video(label="Upload Video"),
          gr.CheckboxGroup(choices=coco_labels[1:], label="Select Classes"),
          gr.Slider(label = "blur intensity",minimum = 3,maximum = 90, step = 3),
          gr.Number(label="Start Time (seconds)"),
          gr.Number(label="End Time (seconds)"),
      ],
      outputs= "video",
      title = 'BlurVista 👓'
  )
  iface.launch(debug =  True,inline = False)

