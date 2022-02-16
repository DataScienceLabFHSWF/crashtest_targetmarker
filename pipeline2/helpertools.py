"""
This code allows for the usage of multiple convenience funtions like the conversion from tf to openCV image handling etc.
It also implements a videoHelper class which helps to keep track of the frames. Additionally it provides the loading and inference
functionalities for various neural networks

@author: Daniel Gierse
"""

# Specify model imports
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
import cv2
import numpy as np
import os
from os.path import isfile,isdir,join

# Disable GPU if necessary
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import warnings
import logging,sys

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
warnings.filterwarnings('ignore')


def convert_coords_tf_to_cv2(bboxTF,frame_width,frame_height): #absolute
    return (int(bboxTF[1]*frame_width),int(bboxTF[0]*frame_height),int((bboxTF[3]-bboxTF[1])*frame_width),int((bboxTF[2]-bboxTF[0])*frame_height))
    
def convert_coords_cv2_to_tf(bboxCV2,frame_width,frame_height): #relative
    return (bboxCV2[1]/frame_height,bboxCV2[0]/frame_width,(bboxCV2[3]+bboxCV2[1])/frame_height,(bboxCV2[2]+bboxCV2[0])/frame_width)

def __get_color(colorcode):
    if colorcode == 0: 
        color = (255,51,255) # pink
    elif colorcode == 1: 
        color = (51,204,0) # green
    elif colorcode == 2: 
        color = (0,102,204) # orange
    elif colorcode == 3: 
        color = (0,255,255) # yellow
    elif colorcode == 4:
        color = (255,0,255) # lila
    elif colorcode == 10:
        color = (0,255,255) # lila
    elif colorcode == 11:
        color = (255,255,0) # lila
    elif colorcode == 12:
        color = (122,55,255) # lila
    else:
        color = (0,255,96)

    return color
    
def draw_circle(image,x,y,radius=2,colorcode=1,thickness=-1):
    color = __get_color(colorcode)
    image = cv2.circle(image, (int(x),int(y)), radius, color, thickness) 
    return image

def draw_point(image,x,y,colorcode=1):
    color = __get_color(colorcode)

    image[y,x] = color
    return image

def save_image(image,filepath,target_region=[]):
    logging.debug("target_region: {}".format(target_region))
    if len(target_region) == 0:
        cv2.imwrite(join(filepath),image)
    else:
        img = image[target_region[1]:(target_region[1]+target_region[3]-1),target_region[0]:(target_region[0]+target_region[2]-1)]
        cv2.imwrite(join(filepath),img)
        

def draw_bounding_box(image,bbox_coordinates_cv2,label,colorcode=1): # [xmin,ymin,width,height] abs
    color = __get_color(colorcode)

    logging.debug("draw_bounding_box: label: {}".format(label))    

    p1 = (int(bbox_coordinates_cv2[0]), int(bbox_coordinates_cv2[1]))
    p2 = (int(bbox_coordinates_cv2[0] + bbox_coordinates_cv2[2]), int(bbox_coordinates_cv2[1] + bbox_coordinates_cv2[3]))
    cv2.rectangle(image, p1, p2, color, 2, 1)
    cv2.putText(image,label,(int(p1[0]),int(p1[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return image
    
def draw_bounding_box_score(image,bbox_coordinates_cv2,label,score=0,colorcode=1): # [xmin,ymin,width,height] abs
    color = __get_color(colorcode)  

    p1 = (int(bbox_coordinates_cv2[0]), int(bbox_coordinates_cv2[1]))
    p2 = (int(bbox_coordinates_cv2[0] + bbox_coordinates_cv2[2]), int(bbox_coordinates_cv2[1] + bbox_coordinates_cv2[3]))
    cv2.rectangle(image, p1, p2, color, 2, 1)
    cv2.putText(image,"{}: {}".format(label,score),(int(p1[0]),int(p1[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return image

def draw_text(img,class_id,x,y,info_data,rows=None,cols=None,section=None):
    if rows is None and cols is None: # absolute coordinates given
        x = x
        y = y
    else: # relative coordinates given, need to multiply by factor to get absolute coordinates for drawing box
        x = x * cols
        y = y * rows

    if class_id == 4:
        cv2.putText(img, '{:2f} cm/pixel'.format(info_data), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), lineType=cv2.LINE_AA)

    elif class_id == 5:
        cv2.putText(img, '{} targetmarker currently active'.format(info_data), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), lineType=cv2.LINE_AA)

    else:
        pass

    return img
      
def calc_iou_tf(box1, box2, iou_thresh):
        ymin_inter = max(box1[0],box2[0])
        xmin_inter = max(box1[1],box2[1])
        ymax_inter = min(box1[2],box2[2])
        xmax_inter = min(box1[3],box2[3])

        box1_area = (box1[3] - box1[1] + 1) * (box1[2] - box1[0] + 1)
        box2_area = (box2[3] - box2[1] + 1) * (box2[2] - box2[0] + 1)

        inter_area = max(0,ymax_inter-ymin_inter+1) * max(0,xmax_inter-xmin_inter+1)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area/union_area
        
        if iou > iou_thresh:
            succ = True
        else:
            succ = False
        
        return succ, iou

def calc_iou_cv2(bbox1, bbox2, iou_thresh=0.4):  # expects [xmin,ymin,width,height] abs, converts to [ymin,xmin,ymax,xmax] abs
    box1 = [bbox1[1],bbox1[0],bbox1[1]+bbox1[3],bbox1[0]+bbox1[2]]
    box2 = [bbox2[1],bbox2[0],bbox2[1]+bbox2[3],bbox2[0]+bbox2[2]]
    
    return calc_iou_tf(box1,box2,iou_thresh)

"""
keeps track of the video, more convenient than cv2 video implementation for this purpose
"""

class VideoOperations:
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.length_in_frames = 0
        self.frame_width = 0
        self.frame_height = 0
        self.video_loaded = False
        self.writing_video = False
        self.cap = None
    
    def load_video(self,path_to_video):
        self.reset()
        if isfile(join(path_to_video)):
            self.cap = cv2.VideoCapture(path_to_video)
            cap_succ, image = self.cap.read()

            if cap_succ:
                self.video_loaded = True
                self.frame_width = int(self.cap.get(3))
                self.frame_height = int(self.cap.get(4))
                self.length_in_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            cap_succ = False
        return cap_succ
    
    # eg vid-length = 421 frames => array = [0 ... 420], highest address = 420, highest frame_id = 421
    def get_frame_at_pos(self,frame_id):
        succ = False
        frame = None
        frame_normalized = frame

        if self.video_loaded:
            if frame_id < 1:
                frame_id = 1
                
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id) - 1) # needs to be set to 1 frame before the actual frame you wanna read (read() takes the next frame -> incrementing before evaluating frame)
            succ, frame = self.cap.read()

            if succ:
                frame_normalized = cv2.normalize(frame, None, 0, 255, norm_type=cv2.NORM_MINMAX)
            else:
                frame_normalized = frame
        
        return succ, frame, frame_normalized
            
    def get_frame(self):
        succ = False
        frame = None
        if self.video_loaded:
            succ, frame = self.cap.read()
            if succ:
#                frame_normalized = cv2.GaussianBlur(cv2.normalize(frame, None, 0, 255, norm_type=cv2.NORM_MINMAX),(3,3),2)
                frame_normalized = cv2.normalize(frame, None, 0, 255, norm_type=cv2.NORM_MINMAX)
            else:
                frame_normalized = frame

        return succ, frame, frame_normalized

    #@profile    
    def get_all_frames(self):
        succ = False
        frames = []
        
        if self.video_loaded:
            self.restart_video()
            read_succ, image = self.cap.read()
            succ = read_succ
            
            while read_succ:
                frames.append(image)
                read_succ, image = self.cap.read()

        return succ, frames
    
    def get_all_frames_sk(path):
        succ = True
        frames = []
        cap = video(path)
        fc = cap.frame_count()
        for i in np.arange(fc):
            frames.append(cap.get_index_frame(i))
        
        return succ, frames
    
    def get_current_frame_pos(self):
        if self.video_loaded:
            return True, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return False, -1
    
    def apply_edge_filter(self,frame):
        if frame is not None:
            return True,cv2.Canny(frame,20,50)
        return False,frame    
                     
    def restart_video(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    """ 
    check whether or not a scale is present in the video
    """
    def check_scale_in_video(self,scalereader):
        has_scale_in_video = False
        frames_percentages_for_target_region_selection = [10,17,22,25,27,29,40,45,50,55]
        
        target_frames = [self.get_frame_at_pos(int(self.length_in_frames*percentage/100))[1] for percentage in frames_percentages_for_target_region_selection]
        
        outer_match_counter = 0
        for frame in target_frames:
            cpp,cpp_avg,cpp_floating_avg,scale_matches,best_ssim = scalereader.read_scale(frame,False)
            
            inner_match_counter = 0
            has_scale_in_frame = False
            for match in scale_matches:
                if match[2] > 0 and match[3] > 0:
                    inner_match_counter += 1
                    
                    if inner_match_counter == 3:
                        has_scale_in_frame = True
                        break
            
            if has_scale_in_frame:
                outer_match_counter += 1
                
                if outer_match_counter == 3:
                    has_scale_in_video = True
        
        logging.debug("has_scale_in_video: {}".format(has_scale_in_video))
        
        self.restart_video()
        
        return has_scale_in_video
    
    """ 
    try to find a bounding box across the whole video that encapsulates the movement of the car
    """
    def find_target_region_bbox(self,obj_detector=None,min_target_score=0.5):
        succ = False
        target_frames = None
        target_region_bbox = [0,0,1,1]
        target_region_bbox_cv2 = [0,0,self.frame_width,self.frame_height]
        frames_for_target_region_selection = [0.1,10,30,50,90,99]

        if obj_detector is not None:  
            target_frames = [self.get_frame_at_pos(int(self.length_in_frames*percentage/100))[1] for percentage in frames_for_target_region_selection]

            """
            https://stackoverflow.com/questions/42376201/list-comprehension-for-multiple-return-function#42376244
            """
            target_succs, target_bboxes_all = zip(*[obj_detector.detect(frame,min_target_score)[0:2] for frame in target_frames])
            
            target_bboxes = []
            
            # check whether detection worked in the first and one of the last frames (to ensure correct min/max values)
            if target_succs[0] and (target_succs[len(frames_for_target_region_selection)-1] or target_succs[len(frames_for_target_region_selection)-2]):
                for i,bboxes_frame in enumerate(target_bboxes_all):
                    if target_succs[i]:
                        target_bboxes.append(bboxes_frame[0])
            
                min_y = np.amin([bbox[0] for bbox in target_bboxes])
                min_x = np.amin([bbox[1] for bbox in target_bboxes])
                max_y = np.amax([bbox[2] for bbox in target_bboxes])
                max_x = np.amax([bbox[3] for bbox in target_bboxes])
                
                target_region_bbox = [min_y,min_x,max_y,max_x]

                target_region_bbox_cv2 = convert_coords_tf_to_cv2(target_region_bbox, self.frame_width, self.frame_height)
                target_region_bbox = [min_x,min_y,max_x,max_y]
                
                succ = True

        self.restart_video()

        return succ, target_region_bbox, target_region_bbox_cv2
        
    def write_video(self,frames,output_folder,vid_name,suffix,fps,width=None,height=None):
        succ = False
        
        if len(frames) > 0 and isdir(output_folder):
            if width is None:
                width = self.frame_width
            if height is None:
                height = self.frame_height
            
            writer = cv2.VideoWriter(join(output_folder, vid_name + suffix),
                                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width,height))
            
            for frame in frames:
                writer.write(frame)
            writer.release()
            
            if isfile(join(output_folder, vid_name + suffix)):
                succ = True
        
        return succ

""""       
 Create object detector. This was taken from github and extended upon
"""
class TFObjectDetector():
  
  # Constructor
      def __init__(self, path_to_object_detection = './models/research/object_detection/configs/tf2',\
        path_to_model_checkpoint = './checkpoint', path_to_labels = './labels.pbtxt',\
          model_name = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8', min_detection_score = 0.8):
        self.model_name = model_name
        self.pipeline_config_path = path_to_object_detection
        self.pipeline_config = os.path.join(f'{self.pipeline_config_path}/{self.model_name}.config')
        self.full_config = config_util.get_configs_from_pipeline_file(self.pipeline_config)
        self.path_to_model_checkpoint = path_to_model_checkpoint
        self.path_to_labels = path_to_labels
        self.setup_model()
        self.tracking = False
        self.min_detection_score = min_detection_score
    
      # Set up model for usage
      def setup_model(self):
        self.build_model()
        self.restore_checkpoint()
        self.detection_function = self.get_model_detection_function()
        self.prepare_labels()
    
      # Build detection model
      def build_model(self):
        model_config = self.full_config['model']
        assert model_config is not None
        self.model = model_builder.build(model_config=model_config, is_training=False)
        return self.model
    
      # Restore checkpoint into model
      def restore_checkpoint(self):
        assert self.model is not None
        self.checkpoint = tf.train.Checkpoint(model=self.model)
        self.checkpoint.restore(os.path.join(self.path_to_model_checkpoint, 'ckpt-0')).expect_partial()
    
      # Get a tf.function for detection
      def get_model_detection_function(self):
        assert self.model is not None
    
        @tf.function
        def detection_function(image):
          image, shapes = self.model.preprocess(image)
          prediction_dict = self.model.predict(image, shapes)
          detections = self.model.postprocess(prediction_dict, shapes)

          return detections, prediction_dict, tf.reshape(shapes, [-1])
    
        return detection_function
    
      # Prepare labels
      # Source: https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_tf2_colab.ipynb
      def prepare_labels(self):
          label_map = label_map_util.load_labelmap(self.path_to_labels)
          categories = label_map_util.convert_label_map_to_categories(
                  label_map,
                  max_num_classes=label_map_util.get_max_label_map_index(label_map),
                  use_display_name=True)
          self.category_index = label_map_util.create_category_index(categories)
          self.label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)
        #   print("label_map_dict: {}".format(self.label_map_dict))
          
      # Get keypoint tuples
      # Source: https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_tf2_colab.ipynb
      def get_keypoint_tuples(self, eval_config):
        tuple_list = []
        kp_list = eval_config.keypoint_edge
        for edge in kp_list:
          tuple_list.append((edge.start, edge.end))
        return tuple_list

  
      # Prepare image
      def prepare_image(self, image):
        return tf.convert_to_tensor(
          np.expand_dims(image, 0), dtype=tf.float32
        )
        
        # Perform detection
      def detect(self, image, label_offset = 1):
        # Ensure that we have a detection function
        assert self.detection_function is not None
        # Prepare image and perform prediction
        image = image.copy()
        image_tensor = self.prepare_image(image)
        detections, predictions_dict, shapes = self.detection_function(image_tensor)
    
        # Use keypoints if provided
        keypoints, keypoint_scores = None, None
        if 'detection_keypoints' in detections:
          keypoints = detections['detection_keypoints'][0].numpy()
          keypoint_scores = detections['detection_keypoint_scores'][0].numpy()
        
        return image
    
        # Predict image from folder
      def detect_image(self, path, output_path):                  
        # Load image
        image = cv2.imread(path)
    
        # Perform object detection and add to output file
        output_file = self.detect(image)[3]
        
        # Write output file to system
        cv2.imwrite(output_path, output_file)
  

class TFCarDetector(TFObjectDetector):
    def detect(self, image, min_detection_score = 0.8, label_offset = 1):
        # print("CALL TO TFCarDetector.detect()")
        # Ensure that we have a detection function
        assert self.detection_function is not None
        car_detected = False
        # Prepare image and perform prediction
        image = image.copy()
        image_tensor = self.prepare_image(image)
        detections, predictions_dict, shapes = self.detection_function(image_tensor)
        
        # [ymin,xmin,ymax,xmax]
        detected_bboxes = detections['detection_boxes'][0].numpy()
        detected_classes = detections['detection_classes'][0].numpy()
        detected_scores = detections['detection_scores'][0].numpy()
        
        car_indices = np.where(detected_classes[np.where(detected_scores>=min_detection_score)]==2)

        if len(car_indices[0]) > 0:
            car_detected = True
        
        # Return the image
        return car_detected,detected_bboxes,detected_scores      # ,image
    
class TFMarkerDetectorv3(TFObjectDetector):
    def detect(self, image, min_detection_score_mxt = 0.4, min_detection_score_dot = 0.4, min_detection_score_quad = 0.2, label_offset = 1):

        # Ensure that we have a detection function
        assert self.detection_function is not None
        mxt_detected = False
        dot_detected = False
        quad_detected = False
        # Prepare image and perform prediction
        image = image.copy()
        image_tensor = self.prepare_image(image)
        detections, predictions_dict, shapes = self.detection_function(image_tensor)
        
        # [ymin,xmin,ymax,xmax]
        detected_bboxes = detections['detection_boxes'][0].numpy()
        detected_classes = detections['detection_classes'][0].numpy()
        detected_scores = detections['detection_scores'][0].numpy()
        

        mxt_indices = np.where(detected_classes[np.where(detected_scores>=min_detection_score_mxt)]==0)
        dot_indices = np.where(detected_classes[np.where(detected_scores>=min_detection_score_dot)]==1)
        quad_indices = np.where(detected_classes[np.where(detected_scores>=min_detection_score_quad)]==2)
        
        mxt_selected_scores = detected_scores[mxt_indices[::-1]]
        dot_selected_scores = detected_scores[dot_indices[::-1]]
        quad_selected_scores = detected_scores[quad_indices[::-1]]

        if len(mxt_indices[0]) > 0:
            mxt_detected = True
            detected_bboxes_mxt = detected_bboxes[mxt_indices]
        else:
            detected_bboxes_mxt = []
            
        if len(dot_indices[0]) > 0:
            dot_detected = True
            detected_bboxes_dot = detected_bboxes[dot_indices]
        else:
            detected_bboxes_dot = []

        if len(quad_indices[0]) > 0:
            quad_detected = True
            detected_bboxes_quad = detected_bboxes[quad_indices]
        else:
            detected_bboxes_quad = []
        # Return the image
        return mxt_detected, detected_bboxes_mxt, mxt_selected_scores, dot_detected, detected_bboxes_dot, dot_selected_scores, quad_detected, detected_bboxes_quad, quad_selected_scores, detected_scores      # ,image
    
class TFMarkerDetectorv4(TFObjectDetector):
    def detect(self, image, min_detection_score_mxtdot = 0.4, min_detection_score_quad = 0.2, label_offset = 1):
        # Ensure that we have a detection function
        assert self.detection_function is not None
        mxtdot_detected = False
        quad_detected = False
        # Prepare image and perform prediction
        image = image.copy()
        image_tensor = self.prepare_image(image)
        detections, predictions_dict, shapes = self.detection_function(image_tensor)
        
        # [ymin,xmin,ymax,xmax]
        detected_bboxes = detections['detection_boxes'][0].numpy()
        detected_classes = detections['detection_classes'][0].numpy()
        detected_scores = detections['detection_scores'][0].numpy()
        
        
        
        mxtdot_selected_scores = detected_classes[np.where(detected_scores>=min_detection_score_mxtdot)]
        quad_selected_scores = detected_classes[np.where(detected_scores>=min_detection_score_quad)]
        
        mxtdot_indices = np.where(detected_classes[np.where(detected_scores>=min_detection_score_mxtdot)]==0)
        quad_indices = np.where(detected_classes[np.where(detected_scores>=min_detection_score_quad)]==1)
        
        mxtdot_selected_scores = detected_scores[mxtdot_indices[::-1]]
        quad_selected_scores = detected_scores[quad_indices[::-1]]
            
        
        logging.debug("detected_scores: {}   mxtdot_selected_scores: {}   quad_selected_scores: {}".format(detected_scores, mxtdot_selected_scores, quad_selected_scores))

        if len(mxtdot_indices[0]) > 0:
            mxtdot_detected = True
            detected_bboxes_mxtdot = detected_bboxes[mxtdot_indices]
        else:
            detected_bboxes_mxtdot = []

        if len(quad_indices[0]) > 0:
            quad_detected = True
            detected_bboxes_quad = detected_bboxes[quad_indices]
        else:
            detected_bboxes_quad = []
        # Return the image
        return mxtdot_detected, detected_bboxes_mxtdot, mxtdot_selected_scores, quad_detected, detected_bboxes_quad, quad_selected_scores, detected_scores

class TFScaleDetector(TFObjectDetector):
   def detect(self, image, min_detection_score = 0.8, label_offset = 1):
       # Ensure that we have a detection function
       assert self.detection_function is not None
       scale_detected = False

       # Prepare image and perform prediction
       image = image.copy()
       image_tensor = self.prepare_image(image)
       detections, predictions_dict, shapes = self.detection_function(image_tensor)
       
       # [ymin,xmin,ymax,xmax]
       detected_bboxes = detections['detection_boxes'][0].numpy()
       detected_classes = detections['detection_classes'][0].numpy()
       detected_scores = detections['detection_scores'][0].numpy()
       
       scale_indices = np.where(detected_classes[np.where(detected_scores>=min_detection_score)]==0)


       if len(scale_indices[0]) > 0:
           scale_detected = True
           detected_bboxes_scale = detected_bboxes[scale_indices]
       else:
           detected_bboxes_scale = []

       # Return the image
       return scale_detected, detected_bboxes_scale, detected_scores
    
class TFMXTDOTDetector(TFObjectDetector):
    def detect(self, image, min_detection_score = 0.6):

        # Ensure that we have a detection function
        assert self.detection_function is not None
        mxt_detected = False
        dot_detected = False
        max_score_dot = 0
        max_score_mxt = 0

        # Prepare image and perform prediction
        image = image.copy()
        image_tensor = self.prepare_image(image)
        detections, _, _ = self.detection_function(image_tensor)
        
        # [ymin,xmin,ymax,xmax]
        detected_bboxes = detections['detection_boxes'][0].numpy()
        detected_classes = detections['detection_classes'][0].numpy()
        detected_scores = detections['detection_scores'][0].numpy()
        
        mxt_indices = np.where(detected_classes[np.where(detected_scores>=min_detection_score)]==0)
        dot_indices = np.where(detected_classes[np.where(detected_scores>=min_detection_score)]==1)

#        best_mxt_score = detected_scores

        if len(mxt_indices[0]) > 0:
            mxt_detected = True
            detected_bboxes_mxt = detected_bboxes[mxt_indices]
            detected_scores_mxt = detected_scores[mxt_indices]
            max_score_mxt = max(detected_scores_mxt)
        else:
            detected_bboxes_mxt = []
            
        if len(dot_indices[0]) > 0:
            dot_detected = True
            detected_bboxes_dot = detected_bboxes[dot_indices]
            detected_scores_dot = detected_scores[dot_indices]
            max_score_dot = max(detected_scores_dot)
        else:
            detected_bboxes_dot = []

        if max_score_dot > max_score_mxt:
            label = "DOT"
        elif max_score_dot < max_score_mxt:
            label = "MXT"
        else:
            label = "unknown"
            
        logging.debug("TFMXTDOTDetector: label = {}".format(label))

        # Return the image
        return label, mxt_detected, detected_bboxes_mxt, dot_detected, detected_bboxes_dot, detected_scores      # ,image
