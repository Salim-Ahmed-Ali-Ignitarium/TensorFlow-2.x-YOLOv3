# Author: Salim
import os
import cv2
import numpy as np
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp, image_preprocess, postprocess_boxes, nms
from yolov3.configs import *     #remove * imports



class VehicleDetector():
	yolo_obj = ""
	def __init__(self, detect_classes=None):
		'''
		YOLOv3 TF2 module class for vehicle inference, training, crop. Supports object both normal and tiny implementations.
		TO DO: 1.add tiny and normal implementations
				2. add choice of detect_classes
		'''
		detect_classes = detect_classes 
		VehicleDetector.yolo_obj = Load_Yolo_model()
	
	def train(self):
		pass

	def predict(self, image_path, pred_result_path):
		bboxes, image = VehicleDetector.get_bounding_boxes(image_path, pred_result_path, input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255,0,0))
		# print(bboxes)
		# out_boxes, out_scores, out_classes, classes, colors, __ = detections
		return self.get_vehicle_crop(image, bboxes)


	def get_bounding_boxes(image_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
		original_image      = cv2.imread(image_path)
		original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
		original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

		image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
		image_data = image_data[np.newaxis, ...].astype(np.float32)

		if YOLO_FRAMEWORK == "tf":
			pred_bbox = VehicleDetector.yolo_obj.predict(image_data)
		elif YOLO_FRAMEWORK == "trt":
			batched_input = tf.constant(image_data)
			result = VehicleDetector.yolo_obj(batched_input)
			pred_bbox = []
			for key, value in result.items():
				value = value.numpy()
				pred_bbox.append(value)
			
		pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
		pred_bbox = tf.concat(pred_bbox, axis=0)
		
		bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
		bboxes = nms(bboxes, iou_threshold, method='nms')
		
		return bboxes, original_image

	def get_vehicle_crop(self, image, bboxes):
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
	def display_image(cls, cv_image, waitkey_param=0):
		cv2.imshow('test', cv_image)
		cv2.waitKey(waitkey_param)
		cv2.destroyAllWindows()	
		
    # def padding():
    #     pass

if __name__=='__main__':
	Available_vehicle_coco_classes = {'car':2, 'bus':5, 'truck':7}
	vehicle_obj = VehicleDetector([2, 5, 7])
	
	image_path   = "./IMAGES/street.jpg"
	pred_result_path = "./IMAGES/street_pred.jpg"
	# video_path   = "./IMAGES/test.mp4"
	
	# test image 1 with time check
	t1 = time.time()
	cropped = vehicle_obj.predict(image_path, pred_result_path)
	print("TIME", time.time()-t1)
	VehicleDetector.display_image(cropped, 0)
	

	# test image 2
	image_path = "./IMAGES/city.jpg"
	pred_result_path = "./IMAGES/city_pred.jpg"
	t1 = time.time()
	cropped = vehicle_obj.predict(image_path, pred_result_path)
	print("TIME", time.time()-t1)
	VehicleDetector.display_image(cropped, 0)


	
