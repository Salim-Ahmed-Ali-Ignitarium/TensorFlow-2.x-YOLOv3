import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp, detect_vehicle
from yolov3.configs import * #remove * imports

image_path   = "./IMAGES/street.jpg"
video_path   = "./IMAGES/test.mp4"


class vehicleDetector():
	def __init__(self, detect_classes=None):
		'''
		YOLOv3 TF2 module class for vehicle inference, training, crop. Supports object both normal and tiny implementations.
		'''
		self.detect_classes = detect_classes # None, #[5, 7] refer coco_classes.txt 'yolo/pretrained_weights/coco_classes.txt' for details
		self.yolo_obj = Load_Yolo_model()
	
	def train(self):
		pass

	def predict(self, image):
		detections = self.yolo_obj.Predict(image)
		out_boxes, out_scores, out_classes, classes, colors, __ = detections
		return self.getVehicleCrop(image, out_boxes)

	def getVehicleCrop(self, image, out_boxes):
		cropped = None
		for bbox in out_boxes:
			if bbox is not None:
				x1, y1, x2, y2 = bbox 
				cropped = image[y1:y2, x1:x2]
				break
		return cropped

    def padding():
        pass

if __name__=='__main__':
	Available_vehicle_coco_classes = {'car':2, 'bus':5, 'truck':7}
	
	vehicle_obj = vehicleDetector([2, 5, 7])
	image = cv2.imread('../testfiles/NTTE-CAM12-20200426_151_0125_73.jpg')
	cropped = vehicle_obj.predict(image)

	#display_image(cropped, waitkey=0)
	
