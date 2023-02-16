# This is the file that implements a flask server to do inferences.
from __future__ import division, print_function, absolute_import

import os
import sys
import cv2
import numpy as np
import json
import time
import uuid
import math

import torch
import pickle
import pytz

import boto3
from ast import literal_eval
from datetime import datetime, timedelta
import flask

from config import *
from tracker import Tracker
from utils.visualize import vis_track,visualize_track_resume

region=os.environ.get("REGION", "us-east-1")
s3_client = boto3.client('s3', region_name=region)
dynamodb = boto3.resource('dynamodb', region_name=region)
out_dir = '/videos/output/'
model_path = "weights/"
app_name="cpg-booth-demo"

class VideoService:  # for inference
    def __init__(self, path):
        if not os.path.isfile(path):
            raise FileExistsError
        
        self.cap = cv2.VideoCapture(path)        
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print('Length of {}: {:d} frames'.format(path, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    def get_VideoLabels(self):
        return self.cap, self.frame_rate, self.vw, self.vh

class TrackList: 
    def __init__(self, models_path, name, zones, start_time,detectors_config, tracker_config): 
        self.camera = name
        self.zones = zones
        self.tracker = Tracker(models_path=models_path, model='yolox-s', detectors_config=detectors_config, tracker_config=tracker_config)

        self.people = dict()
        self.people_results = dict()
        self.current_CB = dict()
        self.last_time_CB = dict()
        self.primitives= dict()
        self.primitives["reaching to shelf"]=[["towards shelf", "out shelf", "out shelf"], ["towards shelf", "in shelf", "in shelf"]]
        self.primitives["retracting from shelf"]=[["leave shelf", "in shelf", "in shelf"], ["leave shelf", "out shelf", "out shelf"]]
        self.primitives["walking"]=[["!stop", "out shelf", "out shelf"]]
        self.primitives["with hand in shelf"]=[["*", "in shelf", "in shelf"]]
        self.primitives["viewing"]=[["stop", "out shelf", "out shelf"]]

        print('---- Tracking Element added ---')
        print('camera: {}'.format(self.camera))
        print('zones: {}'.format(self.zones))

    
class TrackingService(object):
    model_input_size = config['model_input_size']
    timezone = os.environ.get("TIMEZONE", config['timezone'])
    s3_bucket = os.environ.get("S3_BUCKET", config['s3_bucket'])
    refresh_threshold = float(os.environ.get("REFRESH_THRESHOLD", "300")) #Seconds
    refresh_app_time = float(os.environ.get("REFRESH_APP_TIME", "24")) #Hours
    dynamo_table = float(os.environ.get("DYNAMO_TABLE", config['dynamo_table'])) 
    
    frame_rate = 1
    tracking_list = []
    local_videos = []
    videos = []
    cameras = []
    errors = []
    bbox_thickness = 0 
    division_area = 0
    label_coords = []
    initial_time = None
    os.makedirs(out_dir, exist_ok=True)

    @classmethod
    def init_config(self, args):
        try:
            print('Getting parameters')
            self.videos = args.get('videos', [])
            self.cameras = args.get('cameras', [])
            self.division_area = args.get('division_area')
            self.refresh_threshold = args.get('refresh_threshold', self.refresh_threshold )
            self.frame_rate = args.get('frame_rate', self.frame_rate )
            
            config['tracker_config']['threshold'] = int(args.get("tracking_threshold", "70"))
            config['detectors_config']['coco_dataset']['threshold'] =  int(args.get("detection_threshold", "80"))
    
        except:
            print('Error during initialization.')
        finally:
            self.set_initial_time()
            self.define_track_objects()
            
            if self.videos:
                self.download_s3_videos()

            print('Cuda is available: {}'.format(torch.cuda.is_available()))
            print('Detection threshold: {}'.format(config['detectors_config']['coco_dataset']['threshold']))
            print('Tracker threshold: {}'.format(config['tracker_config']['threshold']))
            print('Division Area: {}'.format(self.division_area))
            print('Videos to process: {}'.format(self.videos))
            print('Videos to process (locally): {}'.format(self.local_videos))
            print('Refresh detection threshold: {}'.format(self.refresh_threshold))
            print('Initialiation complete.')
            
    @classmethod
    def define_track_objects(self):
        if os.path.exists('tracklist.pickle'):
            self.tracking_list = self.load_pickle('tracklist')
            print("tracking_list len: {}".format(len(self.tracking_list)))

        registered_cameras = [ tl.camera for tl in self.tracking_list]
        zones = dict()

        for cam in self.cameras:
            if cam["id"] not in registered_cameras:
                for class_def in config['classes_def'].keys():
                    area = config['classes_def'][class_def]['area']
                    zones[area] = {
                        "area": literal_eval(cam[f"{area}_zone"]),
                        "color": literal_eval(cam[f"{area}_zone_color"])
                    }
                    
                self.tracking_list.append(TrackList(model_path, cam["id"], zones, datetime.now(pytz.timezone(self.timezone)), config['detectors_config'], config['tracker_config']))

    @classmethod
    def set_initial_time(self):
        now = datetime.now(pytz.timezone(self.timezone))
        if os.path.exists('apptime.pickle'):
            self.initial_time = self.load_pickle('apptime')
    
        if not self.initial_time:
            self.initial_time = now
            self.save_pickle("apptime", self.initial_time)

        final_date = self.initial_time + timedelta(hours=self.refresh_app_time) 

        if (now >= final_date):
            print('Cleaning tracking list. Now: {} - Expiration Date: {}'.format(now, final_date))
            self.tracking_list = []
            self.initial_time = now
            self.save_pickle('apptime', self.initial_time)   
            self.save_pickle('tracklist', self.tracking_list)
        else: 
            print('Resources Expiration Date: {}'.format(final_date))  

    @classmethod
    def clean_track_history(self, track_obj): 
        people, people_results, current_CB, last_time_CB = dict(),dict(),dict(),dict()

        for people_id in track_obj.last_time_CB:
            if ((datetime.now(pytz.timezone(self.timezone)) - track_obj.last_time_CB[people_id]).total_seconds()) < self.refresh_threshold:
                people[people_id] = track_obj.people[people_id]
                last_time_CB[people_id] = track_obj.last_time_CB[people_id]
                if (track_obj.people_results.get(people_id, None)):
                    people_results[people_id] = track_obj.people_results[people_id]
                if (track_obj.current_CB.get(people_id, None)):
                    current_CB[people_id] = track_obj.current_CB[people_id]
            else:
                print(f'People ID {people_id} deleted from tracking object')
        return people, people_results, current_CB, last_time_CB 


    

    @classmethod
    def download_s3_videos(self):
        self.local_videos = []
        videos_json = self.videos
        for vid in videos_json:
            try:
                complete_path = (vid["path"].split('s3://')[1])
                bucket, video_path = complete_path.split('/',1)  
                video_name = video_path.split('/')[-1] or video_path
                os.makedirs('videos', exist_ok=True)
                video_download_path = 'videos/' + video_name

                try:
                    with open(video_download_path, 'wb') as f:
                        s3_client.download_fileobj(bucket, video_path, f)

                    print('Video "{}" successfully downloaded from Bucket "{}"'.format(video_name, bucket))
                    self.local_videos.append({'video': video_download_path, 'area': vid["area"]})
                except Exception as e:
                    print('Error downloading video "{}" from Bucket "{}"'.format(video_name, bucket))
                    print(e)
            except Exception as e:
                print('S3 video path is not properly formed: {}'.format(vid["path"]))
                print(e)
       
    @classmethod
    def save_pickle(self, filename, object):
        print(f'Save {filename} to pickle')
        with open(f"{filename}.pickle", "wb") as file_:
            pickle.dump(object, file_, -1)

    @classmethod
    def load_pickle(self, filename):
        print(f'Restoring {filename} from pickle')
        sys.path.append('./YOLOX/exps/default')
        return pickle.load(open(f"{filename}.pickle", "rb", -1))
    
    @classmethod
    def preprocess(self, image, input_size, swap=(2, 0, 1)):
        if len(image.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / image.shape[0], input_size[1] / image.shape[1])
        resized_img = cv2.resize(
            image,
            (int(image.shape[1] * r), int(image.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(image.shape[0] * r), : int(image.shape[1] * r)] = resized_img
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r  
    
    @classmethod
    def remove_files(self, files):
        for file in files:
            os.remove(file) 


    @classmethod
    def calculateDistance(self, p1, p2):
        """Calculates and returns the Euclidean distance between two points p1 and p2, which are in format (x1,y1) and (x2,y2) tuples"""
        return math.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))
    
    @classmethod
    def motionEvent(self, p1, p2, distance, threshold, division_area):
        """Returns motion event"""
        #1.calculate direction (degrees) of movement
        direction = math.degrees (math.acos((p2[0]-p1[0])/distance))
        if (p2[1]-p1[1]) < 0:
            direction = 360 - direction

        motion,start_area,end_area = 'stop','',''

        #2.determine motion based on the degrees of movement
        if distance <= threshold:
            motion = 'stop'
        else:
            if (direction >= 210) and (direction <= 350):
                motion = "leave shelf"
            elif (direction >= 10) and (direction <= 170):
                motion = "towards shelf"
            elif (direction > 170) and (direction < 210):
                motion = "left along shelf"
            else:
                motion = "right along shelf"
        
        #3.determine start_area
        start_area = "in shelf" if p1[1] > division_area else "out shelf"

        #4. determine end_area
        end_area = "in shelf" if p2[1] > division_area else "out shelf"
  
        return [motion, start_area, end_area]


    @classmethod
    def patternMatching(self, sequence_primitives, defined_primitives, timeout):
        """
        Match primitives to determine CB
        
        The primitives are matched in a list of indetified primitives in the video vs a list of defined primitives from more recent 
        primitive to oldest primitive in the list. The timeout how many recent sequence primitives are considered for the match.
        """
        sequence_index=len(sequence_primitives) - 1
        defined_index=len(defined_primitives) - 1
        timeout_counter = 0
        first_index=-1
        
        #Check affection of resetting the defined_index to -1 when there isn't a match
        while (sequence_index > -1) and (timeout_counter <= timeout):
            
            match = True
            
            if defined_primitives[defined_index][0] == "!stop":
                if sequence_primitives[sequence_index][0] == "stop":
                    match = False
            elif defined_primitives[defined_index][0] != "*":
                if defined_primitives[defined_index][0] != sequence_primitives[sequence_index][0]:
                    match = False

            if defined_primitives[defined_index][1] != "*":
                if defined_primitives[defined_index][1] != sequence_primitives[sequence_index][1]:
                    match = False

            if defined_primitives[defined_index][2] != "*":
                if defined_primitives[defined_index][2] != sequence_primitives[sequence_index][2]:
                    match = False
                        
            if match:
                defined_index -= 1
                if first_index == -1:
                    first_index = sequence_index
                if defined_index == -1:
                    return [True, first_index]
            else:
                timeout_counter += 1
            sequence_index -= 1
        return [False, sequence_index]
    
    @classmethod 
    def dot_prod_with_shared_start(self, start, end1, end2):
            return (end1[0] - start[0]) * (end2[0] - start[0]) + (end1[1] - start[1]) * (end2[1] - start[1])

    @classmethod 
    def is_inside_rectangle(self, vertices, point):
        return all(self.dot_prod_with_shared_start(vertices[i - 1], v, point) > 0 for i, v in enumerate(vertices))

    @classmethod   
    def is_inside_bbox(self, left, right, top, bottom, area):
        area_details = area
        test_point_left = (left,bottom)
        test_point_right = (right,bottom)
        
        area_left = self.is_inside_rectangle(area_details, test_point_left)
        area_right = self.is_inside_rectangle(area_details, test_point_right)
        
        return True if (area_left or area_right) else False
    
    @classmethod
    def update_actions(self, tracking, tl_area,frame_aux, y_coor=120):
        # threshold_1 is used to not take movements with a distance shorter than it for consideration
        # threshold_2 is used to take movements with a distance shorter than it as a stop state
        # timeout is the amount of primitives to be used for matching customer behaviors
        threshold_1,threshold_2,timeout,valid_detections= 1,5,20,[]
        
        for i in range(len(tracking)):
            box = tracking[i]
            x0, y0, x1, y1, id = float(box[0]), float(box[1]), float(box[2]), float(box[3]), int(box[4])
            if self.is_inside_bbox( x0, x1, y0, y1, tl_area.zones['all']['area']):
                valid_detections.append(box)
                tl_area.last_time_CB[id] = datetime.now(pytz.timezone(self.timezone))
                
                #1. Calculating the horizontal mid point of a detected customer which will be use for movement distance calculations
                mid_point = int(((x1-x0)/2)+x0)  
                motion_event = []
                
                #2. Store new detected customer
                if(id not in tl_area.people):    
                    tl_area.people[id] = [mid_point, y0]
                else:
                    p1=(tl_area.people[id][0], tl_area.people[id][1]) #Last (p1) and current (p2) coordinates of the customer
                    p2=(mid_point, y1)
                    distance = self.calculateDistance(p1, p2)
                    if distance > threshold_1:
                        motion_event = self.motionEvent(p1, p2, distance, threshold_2, self.division_area)                                    
                        tl_area.people[id] = [mid_point, y1] #Store current coordinates of the customer

                        #3.Store current detected primitive (motion_event) of the customer
                        if id not in tl_area.people_results:
                            tl_area.people_results[id] = [[motion_event[0], motion_event[1], motion_event[2]]]
                        else:
                            tl_area.people_results[id].append([motion_event[0], motion_event[1], motion_event[2]])
                        
                        #4. Get all matched customer behaviors in the customer primitives
                        current_behaviors = []
                        current_behaviors_dict = dict()
                        for behavior in tl_area.primitives:
                            result = self.patternMatching(tl_area.people_results[id],tl_area.primitives[behavior],timeout)
                            if result[0]:
                                current_behaviors_dict[behavior]=result[1]
                        
                        #5. Apply rules to determine valid current customer behaviors
                        for cr_behavior in current_behaviors_dict:
                            if ((cr_behavior != "walking") and (cr_behavior != "viewing") and (cr_behavior != "with hand in shelf")) or (((cr_behavior == "walking") or (cr_behavior == "viewing") or (cr_behavior == "with hand in shelf")) and (("reaching to shelf" not in current_behaviors_dict) and ("retracting from shelf" not in current_behaviors_dict))):
                                if (cr_behavior == "reaching to shelf") and (("retracting from shelf" not in current_behaviors_dict) or (current_behaviors_dict["reaching to shelf"] > current_behaviors_dict["retracting from shelf"])):
                                    current_behaviors.append(cr_behavior)
                                if (cr_behavior == "retracting from shelf") and (("reaching to shelf" not in current_behaviors_dict) or (current_behaviors_dict["retracting from shelf"] > current_behaviors_dict["reaching to shelf"])):
                                    current_behaviors.append(cr_behavior)
                                if (cr_behavior == "walking") and (("viewing" not in current_behaviors_dict) or (current_behaviors_dict["walking"] > current_behaviors_dict["viewing"])):
                                    current_behaviors.append(cr_behavior)
                                if (cr_behavior == "viewing") and (("walking" not in current_behaviors_dict) or (current_behaviors_dict["viewing"] > current_behaviors_dict["walking"])):
                                    current_behaviors.append(cr_behavior)
                                if (cr_behavior == "with hand in shelf"):
                                    current_behaviors.append(cr_behavior)
                
                        #6. Join all identified customer behaviors and store them
                        if len(current_behaviors) != 0:
                            tl_area.current_CB[id]='|'.join(current_behaviors)

        y_coor = 120
        for i in range(len(valid_detections)):
            box = valid_detections[i]
            frame_aux = vis_track(frame_aux, box)

            id = int(box[4])
            if tl_area.current_CB.get(id, None):
                text_1, text_2 =  'Customer {:d} is '.format(id),tl_area.current_CB[id]
                txt_size_1, txt_size_2 = cv2.getTextSize(text_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0],cv2.getTextSize(text_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                initial_x = 59 + txt_size_1[0]

                visualize_track_resume(frame_aux, 'Customer {:d} is '.format(id) , (0,0,0), (60, y_coor + txt_size_1[1] + 2), (59, y_coor),(59 + txt_size_1[0] + 2, y_coor + int(1.5*txt_size_1[1])+1))
                visualize_track_resume(frame_aux, tl_area.current_CB[id], (204, 204, 0), (initial_x+1, y_coor + txt_size_2[1] + 2), (initial_x, y_coor), (initial_x + txt_size_2[0] + 2, y_coor + int(1.5*txt_size_2[1])+1))
                y_coor += 20

    @classmethod 
    def push_results_to_cloud(self, video_local_path):
        print('Before sending results to cloud')
        table_logs = []
        #Send video to s3
        timestamp = int(time.time())
        now = datetime.now(pytz.timezone(self.timezone))
        video_name = video_local_path.split('/')[-1]
        key = "{}/results/{}_{}_{}_{}_{}_{}".format(app_name, now.month,now.day,now.hour, now.minute,timestamp,video_name)
        
        s3_client.upload_file(video_local_path, self.s3_bucket,key , ExtraArgs={
            'ACL': 'public-read',
            'ContentType': 'video/mp4'
            })
        table_logs.append({'date': now,'video': f"https://{self.s3_bucket}.s3.amazonaws.com/{key}"})

        table = dynamodb.Table(self.dynamo_table)
        with table.batch_writer() as batch:
            for log in table_logs:
                batch.put_item(
                    Item={
                        "id": uuid.uuid4().hex,
                        "date": log['date'].strftime("%Y-%m-%d %H:%M:%S.%f"),
                        "video": log.get('video', '')
                    }
                )

        result = 'Results uploaded to Bucket {}'.format(self.s3_bucket)
        return result
    
    @classmethod
    def process_videos(self, data):
        print(data)
        self.init_config(data)
        if self.local_videos == []:
            print('There are no videos to process')
            json_response = json.dumps({'videos_processed': len(self.local_videos), 'errors': self.errors})
            return json_response
        
        t1 = time.time() 
        s3_response = []
        for video in self.local_videos:
            video_name, video_area = video["video"],video["area"]
            loadvideo = VideoService(video_name)
            video_capture, frame_rate, width, height = loadvideo.get_VideoLabels()
            video_id = video_name.rsplit("/",1)[-1] or video
            video_complete_path = out_dir + (video_id.split('.')[0] or video_id)+'.mp4'
            out_video = cv2.VideoWriter(video_complete_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

            tl_area = None
            for tl in self.tracking_list:
                if tl.camera == video_area:
                    tl_area = tl
                    break 

            frame_cnt,last_frame, first_frame = 0,None,True
            tl_area.people, tl_area.people_results, tl_area.current_CB, tl_area.last_time_CB = self.clean_track_history(tl_area)
            
            while True:
                ret, frame = video_capture.read() 
                if not ret:
                    break
                
                if (frame_cnt % self.frame_rate == 0):
                    print('@@@@@@@@@@@@@@@@@ PROCESSING MEDIA ({}) {}/{} @@@@@@@@@@@@@@@@@'.format(video_name, frame_cnt, int(loadvideo.vn)))
                    ttframe = time.time() 
                    original_frame = frame.copy()
                    frame_aux = frame.copy()
                    cv2.line(frame_aux, (0,self.division_area), (frame_aux.shape[1], self.division_area), (255, 255, 255),3)

                    image_data, ratio = self.preprocess(original_frame, self.model_input_size)  
                    tracking, dets = tl_area.tracker.update(original_frame, image_data, ratio)
                    visualize_track_resume(frame_aux, f'Total amount of customers on screen: {len(tracking)}')

                    if len(tracking) > 0:
                        self.update_actions(tracking, tl_area, frame_aux)
                    out_video.write(frame_aux)
                    last_frame,first_frame = frame_aux, False
                    print(tl_area.people)
                    print('Frame processing time', time.time() - ttframe)
                else:
                    out_video.write(frame if first_frame else frame_aux) 
                frame_cnt += 1
                 
                
            print('tl_area size: {}'.format(sys.getsizeof(tl_area)))
            out_video.release()
            video_capture.release()
            s3_response.append(self.push_results_to_cloud(video_complete_path))
            self.remove_files([video_complete_path, video_name])

            print('Local results saved at: Video -> {}'.format(video_complete_path))
        
        print('Total time of execution: {} seconds'.format(int(time.time() - t1)))
        return json.dumps({'videos_processed': len(self.local_videos), 's3_results': s3_response, 'errors': self.errors})

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
#     health = TrackingService.get_model() is not None  # You can insert a health check here

#     status = 200 if health else 404
    status = 200 

    return flask.Response(response="\n", status=status, mimetype="application/json")

@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    print("==================================")
    print("Endpoint invoked")
    
    # Do the prediction
    result = TrackingService.process_videos(json.loads(flask.request.data))
    return flask.Response(response=result, status=200, mimetype="application/json")