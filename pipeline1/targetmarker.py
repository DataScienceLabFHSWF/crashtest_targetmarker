"""
This class is used to deal with the targetmarkers. It is designed to handle its own status through methods like update(), confirm() etc
so that it can handle itself instead of always having to keep track on it in the main program. It also offers the classificiation by a Bayesian Neural 
Network and adds filter functionality as well as the center point detection for DOT and MXT markers that was talked about in the bachelor thesis.

@autor Daniel Gierse
"""

import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2 as cv
from os import listdir
from os.path import isfile, join
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import scipy.stats as stats
import helpertools as htools
import itertools
from skimage import measure
from imutils import contours
import logging


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

"""
targetmarker_class_id:
 1: DOT
 2: QUAD
 3: MXT

 bbox format:   tf: [ymin,xmin,ymax,xmax] relative (0-1)
                cv2:[xmin,ymin,width,height] absolute
"""
class Targetmarker:
    newest_id = itertools.count()
    
    mxt_list = []
    dot_list = []
    marker_list = []
    mxt_median_area = 0
    dot_median_area = 0
    min_area_threshold = 0
    
    @classmethod
    def on_marker_append(cls,marker_instance):
        cls.marker_list.append(marker_instance)
#        print("on marker append -  len(marker_list): {}".format(len(cls.marker_list)))
        
    @classmethod
    def on_frame_end(cls):
        cls.mxt_median_area = np.median([marker.bboxes_cv2[-1][2]*marker.bboxes_cv2[-1][3] for marker in cls.marker_list if (marker.is_active and marker.targetmarker_class_id == 0)])
        cls.dot_median_area = np.median([marker.bboxes_cv2[-1][2]*marker.bboxes_cv2[-1][3] for marker in cls.marker_list if (marker.is_active and marker.targetmarker_class_id == 1)])
        
        
        min_mxt = np.min([cls.mxt_median_area])
        min_dot = np.min([cls.dot_median_area])
        
        if math.isnan(min_mxt):
            min_mxt = 0
        if math.isnan(min_dot):
            min_dot = 0
            
        if min_mxt == 0 or min_dot == 0:
            min_median = np.max([min_mxt,min_dot])
        else:
            min_median = np.min([min_mxt,min_dot])
            
        print("min: {}".format(min_median))
        
#        if not math.isnan(min_median):
        cls.min_area_threshold = int(min_median*0.6)
            
        mxts = [marker.bboxes_cv2[-1][2]*marker.bboxes_cv2[-1][3] for marker in cls.marker_list if (marker.is_active and marker.targetmarker_class_id == 0)]
        dots = [marker.bboxes_cv2[-1][2]*marker.bboxes_cv2[-1][3] for marker in cls.marker_list if (marker.is_active and marker.targetmarker_class_id == 1)]

#        cls.mxt_median_area = np.median([marker.target_region_bboxcv2[0]*marker.target_region_bboxcv2[1] for marker in cls.marker_list if (marker.is_active and marker.targetmarker_class_id == 0)])
#        cls.dot_median_area = np.median([marker.target_region_bboxcv2[0]*marker.target_region_bboxcv2[1] for marker in cls.marker_list if (marker.is_active and marker.targetmarker_class_id == 1)])        
#        print("mxt_median: {}     dot_median: {}".format(cls.mxt_median_area,cls.dot_median_area))
#        print("len marker list: {}  mxt_median: {}     dot_median: {}".format(len(cls.marker_list), cls.mxt_median_area,cls.dot_median_area))
#        print("len marker list: {}  len(mxts for median calculation): {}     len(dots for median calculation): {}".format(len(cls.marker_list),len(mxts),len(dots)))
        print("mxts: {} mxt_median: {}".format(mxts,cls.mxt_median_area))
        print("dots: {} dot_median: {}".format(dots,cls.dot_median_area))
        print("cls.min_area_threshold: {}".format(cls.min_area_threshold))
        
    @classmethod
    def reset(cls):
        cls.newest_id = itertools.count()
        cls.marker_list.clear()
        cls.min_area_threshold = 0
        cls.mxt_median_area = 0
        cls.dot_median_area = 0
        
    
    def __init__(self,frame,frame_id,model,bbox,target_region_cv2,bbox_format="tf",targetmarker_class_id=1,confirm_delta_thresh=15,num_bnn_samples=15,bnn_resize_size=32,output_folder="opf",video_id=1,initial_rcnn_score=0,minimum_type_decision_steps_threshold=5,maximum_type_decision_steps_threshold=30):
        
        Targetmarker.on_marker_append(self)#.marker_list.append(self)
        new_id = next(Targetmarker.newest_id)
#        print("create new marker: {}".format(new_id))
        self.marker_id = new_id
        self.frame_width = frame.shape[1]
        self.frame_height = frame.shape[0]
        self.bboxes_cv2 = []
        self.centerpoints_abs = [] #[x,y]
        self.centerpoints_rel = [] 
        self.bnn_predictions = []
        self.reference_centerpoints = []
        self.bbox_centerpoints_abs = []
        self.bbox_centerpoints_rel = []
        self.bbox_centerpoints_rel_vars = []
        self.target_region_bboxcv2 = target_region_cv2
        
        self.start_frame_id = frame_id
        self.video_id = video_id
        
        if bbox_format == "tf":
            self.bboxes_cv2.append(htools.convertCoordsTFtoCV2abs(bbox,self.frame_width,self.frame_height))
        else: # cv2abs
            self.bboxes_cv2.append(bbox)
        
        self.tm_format = targetmarker_class_id
        
        self.last_confirm_delta = 0
        self.last_confirm_delta_thresh = confirm_delta_thresh
        self.confirmed_for_frame = True
        self.is_active = True
        self.is_type_decided = False
        self.type_decision_counter = 0
#        self.minimum_type_decision_steps_threshold = 5
#        self.maximum_type_decision_steps_threshold = 35
        self.minimum_type_decision_steps_threshold = minimum_type_decision_steps_threshold
        self.maximum_type_decision_steps_threshold = maximum_type_decision_steps_threshold
        
        self.bnn_model = model
        self.num_bnn_samples = num_bnn_samples
        self.num_bnn_classes = 2
        self.bnn_multiple_samples = True
        self.bnn_resize_size = bnn_resize_size
        self.plot_loc = output_folder
        
        self.error_intersect = 0.3
        self.starting_variance_x = 10
        self.starting_variance_y = 10
        
        self.rcnn_scores = [initial_rcnn_score]
        
        self.min_avg_prob_threshold = 0.75
        
        self.type_decision_locked = False
        self.type_decision_lock_counter = 0
        self.type_decision_lock_counter_max = self.maximum_type_decision_steps_threshold
        
        
        self.targetmarker_class_id = targetmarker_class_id
        self.is_type_decided = True
        self.__create_tracker(frame)
        self.__update_centerpoint()
        
        self.__update_targetmarker_label()
      
    #
    #
    # updates the printlabel variable which is used for the on-frame drawings:
    #
    # 0: MXT
    # 1: DOT
    # 2: QUAD
    # 3: unknown (temporary)
    # 4: hard to detect
    # 5: better bb found, recheck
    def __update_targetmarker_label(self):
        self.printlabel = "MXTDOT"
        if self.targetmarker_class_id == 0:
            self.printlabel = "MXT"
        elif self.targetmarker_class_id == 1:
            self.printlabel = "DOT"
        elif self.targetmarker_class_id == 2:
            self.printlabel = "QUAD"
        elif self.targetmarker_class_id == 3:
            self.printlabel = "UNKNOWN"
        elif self.targetmarker_class_id == 4:
            self.printlabel = "TOUGH DECISION"
        elif self.targetmarker_class_id == 5:
            self.printlabel = "RECHECKING"
        elif self.targetmarker_class_id == 12:
            self.printlabel = "SKIPPED"


    #
    # Creates new cv2 tracker object and changes class parameters to represent the current status    
    #       
    def __create_tracker(self,frame):
        if self.check_if_in_region(self.bboxes_cv2[-1]):
            tmp_tracker = cv.TrackerCSRT_create()  # für openCV > 4.5.5
            
            logging.debug("__create_tracker -> self.bbox_cv2: {}".format(self.bboxes_cv2[-1]))
            succ = tmp_tracker.init(frame,tuple(self.bboxes_cv2[-1]))
            if succ:
                self.tracker = tmp_tracker
                self.has_tracker_set = True
                self.is_active = True
                print("success: __create_tracker -> self.bbox_cv2: {}".format(self.bboxes_cv2[-1]))
            else:
                self.has_tracker_set = False
                self.is_active = False
        else:
            self.has_tracker_set = False
            self.is_active = False
            
            
    #
    # Checks whether or not the marker lies inside target region        
    def check_if_in_region(self,tm_bboxcv2):
        bbox_in_target_region = False
        if tm_bboxcv2[0] > self.target_region_bboxcv2[0] and tm_bboxcv2[1] > self.target_region_bboxcv2[1] and (tm_bboxcv2[0] + tm_bboxcv2[2]) < (self.target_region_bboxcv2[0] + self.target_region_bboxcv2[2]) and (tm_bboxcv2[1] + tm_bboxcv2[3]) < (self.target_region_bboxcv2[1] + self.target_region_bboxcv2[3]):
            bbox_in_target_region = True
        
        return bbox_in_target_region
    
    #
    # called on the begin of each frame
    # manages the "update cycle" of the marker instance by updating the tracker, checking for the region as well as calling the 
    # prediction and center point methods.
    # On fail the marker gets deactived and/or deleted
    def update(self,frame,frame_normalized,frame_id):
        if self.is_active:
            succ, bbox = self.tracker.update(frame_normalized)
            bbox = np.array(bbox).clip(min=0).astype(np.int)   # tracker sometimes returns negative results (-0.0x) which lead to errors in marker_region assignment
            bbox = tuple([bbox[e] for e in range(len(bbox))])
            
            
            #
            # check whether or not detected tracker was updated succesfully and then check if marker lies inside target region. 
            if succ:
                succ = self.check_if_in_region(bbox)
           
            #    
            # depending on tracker update and position of marker. On update appends latest bounding box to the BB list for this instance.
            # Then extracts the marker region from the image, resizes it to the shape of the Bayesian Neural Network and calls self.__update_model_pred_mxtdot()
            # in order to update the markers prediction whenever it is not classified as DOT marker. Then it calls the functions to update the center points.

            if succ:
                self.bboxes_cv2.append(bbox)
                marker_region = frame[int(bbox[1]):int((bbox[1]+bbox[3])), int(bbox[0]):int((bbox[0]+bbox[2])), :]
                if marker_region.size != 0:
#                    marker_region_resized = cv.resize(marker_region,(self.bnn_resize_size,self.bnn_resize_size), interpolation = cv.INTER_AREA)
#                    if self.targetmarker_class_id != 2:                
#                        self.__update_model_pred_mxtdot(marker_region_resized,frame_id)
                    
                    if self.is_active:
                        self.__update_centerpoint()
#                        centerpoints_intersect_abs = self.__find_centerpoints_intersection()
#                        logging.debug("centerpoints_intersect_abs: {}   centerpoints_intersect_abs[0]: {}    centerpoints_intersect_abs[1]: {}".format(centerpoints_intersect_abs,centerpoints_intersect_abs[0],centerpoints_intersect_abs[1]))
#                        self.centerpoints_abs.append([centerpoints_intersect_abs[0],centerpoints_intersect_abs[1]])
                #
                #
                # deactivates and deletes marker whenever area == 0 (maybe bug)
                else:
                    self.is_active = False
                    del self.bboxes_cv2[-len(self.bboxes_cv2):]
            else:
                logging.debug("TRACKER UPDATE FAILED!")
                self.is_active = False
            
                
    def change_bounding_box(self,frame,bboxcv2,score,frame_id,label_id):
        # Updates the bounding box whenever the detection score of the Faster-RCNN is greater than the markers current one (stored in rcnn_scores).
        # Substitutes latest cv2 BB with the updated one from the RCNN network (tracker update happens "earlier" in the frame than the Faster-RCNN inference)
        # Deletes current tracker and creates new one with the updated coordinates
        # Resets the marker type, as a better fit of the BB should lead to a better prediction of the BNN (and resets decision_counter to allow new prediction process)
        # Reset center point variance since the position inside the BB most likely changed
        # Set ID to 5 ("RECHECKING") in order to visualize the change
#        if self.has_tracker_set:
        del self.tracker
    
        self.bboxes_cv2[-1] = bboxcv2
        self.rcnn_scores.append(score)
        
        self.__create_tracker(frame)
        
        self.targetmarker_class_id = label_id
        self.__update_targetmarker_label()
        

        if self.start_frame_id != frame_id:
            print("change_bounding_box frame_id: {}   start_frame_id: {}".format(frame_id,self.start_frame_id))
            logging.debug("VARIANCE UPDATED")
        else:
            print("ALARM ALARM change_bounding_box frame_id: {}   start_frame_id: {} ALARM".format(frame_id,self.start_frame_id))
        
        
        
    
    def __find_centerpoints_intersection(self):
        return [self.bboxes_cv2[-1][0]+int(self.bboxes_cv2[-1][2]/2),self.bboxes_cv2[-1][1]+int(self.bboxes_cv2[-1][3]/2)]
    
    def __update_centerpoint(self):
        self.centerpoints_abs.append(self.__find_centerpoints_intersection())
    
    def check_confirmation(self):
        if self.confirmed_for_frame:
            self.last_confirm_delta = 0
        else:
            self.last_confirm_delta += 1
            if self.last_confirm_delta > self.last_confirm_delta_thresh:
                self.is_active = False
                # delete unconfirmed entries if there are too many (tracker likely wrong cause of occlusion or stuff)
                del self.bboxes_cv2[-(self.last_confirm_delta + 1):] # +1 cause first_frame (pickup frame) always validated, but wrong in this case 
        self.confirmed_for_frame = False
    
    def confirm(self):
        if self.is_active:
            self.confirmed_for_frame = True
            
