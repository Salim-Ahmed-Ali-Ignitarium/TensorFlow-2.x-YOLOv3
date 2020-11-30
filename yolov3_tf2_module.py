# Author: Salim
import os
import cv2
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp, get_bounding_boxes
from yolov3.configs import *     #remove * imports



class VehicleDetector():
	def __init__(self, detect_classes=None):
		'''
		YOLOv3 TF2 module class for vehicle inference, training, crop. Supports object both normal and tiny implementations.
		'''
		self.detect_classes = detect_classes # None, #[5, 7] refer coco_classes.txt 'yolo/pretrained_weights/coco_classes.txt' for details
		self.yolo_obj = Load_Yolo_model()
	
	def train(self):
		pass

	def predict(self, image):
		bboxes, image = get_bounding_boxes(self.yolo_obj, image_path, "./IMAGES/street_pred.jpg", input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255,0,0))
		print(bboxes)
		# out_boxes, out_scores, out_classes, classes, colors, __ = detections
		return self.getVehicleCrop(image, bboxes)

	def getVehicleCrop(self, image, bboxes):
		cropped = None
		for i, bbox in enumerate(bboxes):
			if bbox is not None:
				coor = np.array(bbox[:4], dtype=np.int32)
				score = bbox[4]
				class_ind = int(bbox[5])
				(x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
				cropped = image[y1:y2, x1:x2]
		return cropped

	@classmethod
	def display_image(cls, cv_image, waitkey_param):
		cv2.imshow('test', cv_image)
		cv2.waitKey(waitkey_param)

    # def padding():
    #     pass

if __name__=='__main__':
	Available_vehicle_coco_classes = {'car':2, 'bus':5, 'truck':7}
	
	vehicle_obj = VehicleDetector([2, 5, 7])
	image_path   = "./IMAGES/street.jpg"
	video_path   = "./IMAGES/test.mp4"
	image = cv2.imread(image_path)
	cropped = vehicle_obj.predict(image)

	VehicleDetector.display_image(cropped, waitkey_param=0)

	
