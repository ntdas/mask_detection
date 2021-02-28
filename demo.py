# import dependencies
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import html
import time
import dlib
import matplotlib.pyplot as plt
import imutils
from imutils.video import FPS
import random
import darknet
from darknet_images import convert2relative
import argparse

# Add command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weight", required=True,
				help="Path to model weights")
ap.add_argument("-c", "--config_file", required=True,
				help="Path to config file")
ap.add_argument("-d", "--data_file", required=True,
				help="Path to data file")
ap.add_argument("-i", "--input", type=str, 
				help="Path to input video")
ap.add_argument("-o", "--output", type=str, 
				help="Path to output video")
ap.add_argument("-s", "--skip_frame", type=int, default=30,
				help="# of skip frames between detections")
ap.add_argument("-t", "--tracking", default=True,
				help="Tracking or not")
args = vars(ap.parse_args())

# configs
# CONFIG_FILE = './yolov4-tiny-obj.cfg'
# DATA_FILE = './obj.data'
# WEIGHTS = './backup/yolov4-tiny-obj_best.weights'

# function to convert the JavaScript object into an OpenCV image
def js_to_image(js_reply):
	"""
	Params:
		js_reply: JavaScript object containing image from webcam
	Returns:
		img: OpenCV BGR image
	"""
	# decode base64 image
	image_bytes = b64decode(js_reply.split(',')[1])
	# convert bytes to numpy array
	jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
	# decode numpy array into OpenCV BGR image
	img = cv2.imdecode(jpg_as_np, flags=1)

	return img

# function to convert OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video stream
def bbox_to_bytes(bbox_array):
	"""
	Params:
		bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.
	Returns:
		bytes: Base64 image byte string
	"""
	# convert array into PIL image
	bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
	iobuf = io.BytesIO()
	# format bbox into png for return
	bbox_PIL.save(iobuf, format='png')
	# format return string
	bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))

	return bbox_bytes

# JavaScript to properly create our live video stream using our webcam as input
def video_stream():
	js = Javascript('''
    var video;
    var div = null;
    var stream;
    var captureCanvas;
    var imgElement;
    var labelElement;
    
    var pendingResolve = null;
    var shutdown = false;
    
    function removeDom() {
       stream.getVideoTracks()[0].stop();
       video.remove();
       div.remove();
       video = null;
       div = null;
       stream = null;
       imgElement = null;
       captureCanvas = null;
       labelElement = null;
    }
    
    function onAnimationFrame() {
      if (!shutdown) {
        window.requestAnimationFrame(onAnimationFrame);
      }
      if (pendingResolve) {
        var result = "";
        if (!shutdown) {
          captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
          result = captureCanvas.toDataURL('image/jpeg', 0.8)
        }
        var lp = pendingResolve;
        pendingResolve = null;
        lp(result);
      }
    }
    
    async function createDom() {
      if (div !== null) {
        return stream;
      }

      div = document.createElement('div');
      div.style.border = '2px solid black';
      div.style.padding = '3px';
      div.style.width = '100%';
      div.style.maxWidth = '600px';
      document.body.appendChild(div);
      
      const modelOut = document.createElement('div');
      modelOut.innerHTML = "<span>Status:</span>";
      labelElement = document.createElement('span');
      labelElement.innerText = 'No data';
      labelElement.style.fontWeight = 'bold';
      modelOut.appendChild(labelElement);
      div.appendChild(modelOut);
           
      video = document.createElement('video');
      video.style.display = 'block';
      video.width = div.clientWidth - 6;
      video.setAttribute('playsinline', '');
      video.onclick = () => { shutdown = true; };
      stream = await navigator.mediaDevices.getUserMedia(
          {video: { facingMode: "environment"}});
      div.appendChild(video);

      imgElement = document.createElement('img');
      imgElement.style.position = 'absolute';
      imgElement.style.zIndex = 1;
      imgElement.onclick = () => { shutdown = true; };
      div.appendChild(imgElement);
      
      const instruction = document.createElement('div');
      instruction.innerHTML = 
          '<span style="color: red; font-weight: bold;">' +
          'When finished, click here or on the video to stop this demo</span>';
      div.appendChild(instruction);
      instruction.onclick = () => { shutdown = true; };
      
      video.srcObject = stream;
      await video.play();

      captureCanvas = document.createElement('canvas');
      captureCanvas.width = 640; //video.videoWidth;
      captureCanvas.height = 480; //video.videoHeight;
      window.requestAnimationFrame(onAnimationFrame);
      
      return stream;
    }
    async function stream_frame(label, imgData) {
      if (shutdown) {
        removeDom();
        shutdown = false;
        return '';
      }

      var preCreate = Date.now();
      stream = await createDom();
      
      var preShow = Date.now();
      if (label != "") {
        labelElement.innerHTML = label;
      }
            
      if (imgData != "") {
        var videoRect = video.getClientRects()[0];
        imgElement.style.top = videoRect.top + "px";
        imgElement.style.left = videoRect.left + "px";
        imgElement.style.width = videoRect.width + "px";
        imgElement.style.height = videoRect.height + "px";
        imgElement.src = imgData;
      }
      
      var preCapture = Date.now();
      var result = await new Promise(function(resolve, reject) {
        pendingResolve = resolve;
      });
      shutdown = false;
      
      return {'create': preShow - preCreate, 
              'show': preCapture - preShow, 
              'capture': Date.now() - preCapture,
              'img': result};
    }
    ''')

	display(js)
 
# Open webcam for video streaming
def video_frame(label, bbox):
	data = eval_js('stream_frame("{}", "{}")'.format(label, bbox))
	return data

# Detect object
def image_detection(image, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

def from_file(network, class_names, class_colors, args):
	vs = cv2.VideoCapture(args["input"])
	totalFrame = 0 
	writer = None
	# fps = FPS().start()

	while True:
		# Start counter
		prev_time = time.time()
		# Read video frame
		frame = vs.read()
		frame = frame[1]
		if frame is None:
			break

		# Convert from BGR to RGB and get shape
		# img = imutils.resize(frame, width=500)
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		H,W = img.shape[:2]

		# create image writer
		if writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)
	  

		# prev_time = time.time()
		if totalFrame % args["skip_frame"] == 0:
			# set the status and initialize our new set of object trackers
			status = "Detecting"
			trackers = []
			labels = []
			confidences = []

			# Object detect
			image, detections = image_detection(img, network, class_names, class_colors, thresh=.25)

			for label, confidence, bbox in detections:
				(x,y,w,h) = convert2relative(image, bbox)
				box = np.array([x-w/2,y-h/2,w,h])*np.array([W,H,W,H])
				(x,y,w,h) = box.astype('int')
				# cv2.rectangle(frame, (x,y), (x+w,y+h), colors[label], 2)
				# cv2.putText(frame, '{} [{:.2f}]'.format(label_names[label], float(confidence)), (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 2)
				# construct a dlib rectangle object from the bounding
				# box coordinates and then start the dlib correlation
				# tracker
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(x, y, x+w, y+h)
				tracker.start_track(img, rect)
				# add the tracker to our list of trackers so we can
				# utilize it during skip frames
				trackers.append(tracker)
				labels.append(label)
				confidences.append(confidence)
		else:
			# loop over the trackers
			for (label, confidence, tracker) in zip(labels, confidences, trackers):
				# set the status of our system to be 'tracking' rather
				# than 'waiting' or 'detecting'
				status = "Tracking"
				# update the tracker and grab the updated position
				tracker.update(img)
				pos = tracker.get_position()
				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())
				# # add the bounding box coordinates to the rectangles list
				# rects.append((startX, startY, endX, endY))
				# bbox_array = cv2.rectangle(bbox_array, (startX, startY), (endX, endY), (0,255,0), 2)
				cv2.rectangle(frame, (startX,startY), (endX,endY), colors[label], 2)
				cv2.putText(frame, '{} [{:.2f}]'.format(label_names[label], float(confidence)), (startX,startY-5), 
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 2)
		totalFrame += 1

		# Display status
		cv2.putText(frame, status, (10, H - 20),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		cv2.putText(frame, 'FPS: {:.2f}'.format(1/(time.time()-prev_time)), (10, H - 40), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

		if writer is not None:
			writer.write(frame)

		
		# fps.update()

	# stop timer and display fps information
	# fps.stop()
	# print('[INFO] total frame: ', totalFrame)
	# print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	if writer is not None:
		writer.release()

	vs.release()

def from_camera_notracking(network, class_names, class_colors, args):
	# start streaming video from webcam
	video_stream()
	# label for video
	label_html = 'Capturing...'
	# initialze bounding box to empty
	bbox = ''
	totalFrame = 0 
	while True:
		# prev_time = time.time()
		js_reply = video_frame(label_html, bbox)
		if not js_reply:
			break

		# convert JS response to OpenCV Image
		# prev_time = time.time()
		img = js_to_image(js_reply["img"])

		# Convert from BGR to RGB and get shape
		prev_time = time.time()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		H,W = img.shape[:2]

		# create transparent overlay for bounding box
		bbox_array = np.zeros([480,640,4], dtype=np.uint8)

		# Object detect
		image, detections = image_detection(img, network, class_names, class_colors, thresh=.25)

		for label, confidence, bbox in detections:
			(x,y,w,h) = convert2relative(image, bbox)
			box = np.array([x-w/2,y-h/2,w,h])*np.array([W,H,W,H])
			(x,y,w,h) = box.astype('int')
			cv2.rectangle(bbox_array, (x,y), (x+w,y+h), colors[label], 2)
			cv2.putText(bbox_array, '{} [{:.2f}]'.format(label_names[label], float(confidence)), (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 2)
	    
		cv2.putText(bbox_array, 'FPS: {:.2f}'.format(1/(time.time()-prev_time)), (10, H - 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)   
		
		
		# convert overlay of bbox into bytes
		bbox_array[:,:,3] = (bbox_array.max(axis = 2) > 0 ).astype(int) * 255
		bbox_bytes = bbox_to_bytes(bbox_array)
		# update bbox so next frame gets new overlay
		bbox = bbox_bytes


if __name__ == '__main__':
	# Load model and determine bbox color
	random.seed(7)
	network, class_names, class_colors = darknet.load_network(config_file=args["config_file"],
															data_file=args["data_file"],
															weights=args["weight"])

	# Initialize color and name for each labels
	colors = {
		'0': (255,0,0),
		'1': (0,255,0),
		'2': (0,0,255)
	}

	label_names = {
		'0': 'no_mask',
		'1': 'mask',
		'2': 'incorrect'
	}

	# Check if using webcam or using recorded video
	if args.get("input", False):
		from_file(network, class_names, class_colors, args)
	elif args.get("tracking"):
		from_camera_notracking(network, class_names, class_colors, args)