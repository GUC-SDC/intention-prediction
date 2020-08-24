import track_updated as tu # tracking
import run_webcam as wc # bodypose 
#import os 
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import cv2
#from head_orientation_keras import head_pose as hp
from keras.models import model_from_json
#import yolotest as yt
import math
import constant_Acc as CA # constant velocity
import time
import json
import ped_direction #pedestrian movement axis FSM
import test_direct as td # image parallelization

#from head_pose  import detect_imgs as Head
import draw_path as draw #draw pedestrian path 

output = open("output.txt","a") 
fp = open('average_values.json',"r")
body_pose_average = json.load(fp)
#bodypose thesholds for each body part (average deltas)
min_right_curb_tier5 = body_pose_average['min_right_curb_tier5']
min_right_curb_tier6 = body_pose_average['min_right_curb_tier6']
min_right_curb_tier7 = body_pose_average['min_right_curb_tier7']
min_left_curb_tier5 = body_pose_average['min_left_curb_tier5']
min_left_curb_tier6 = body_pose_average['min_left_curb_tier6']
min_left_curb_tier7 = body_pose_average['min_left_curb_tier7']
right_curb_tier5 = body_pose_average['right_curb_tier5']
right_curb_tier6 = body_pose_average['right_curb_tier6']
right_curb_tier7 = body_pose_average['right_curb_tier7']
left_curb_tier5 = body_pose_average['left_curb_tier5']
left_curb_tier6 = body_pose_average['left_curb_tier6']
left_curb_tier7 = body_pose_average['left_curb_tier7']
# 
class Bodypart:

    part_id = 0 
    pre_part_x = 0
    pre_part_y = 0
    updated = False
    backup_part_x =0
    backup_part_y =0
    delta_part_x=0
    delta_part_y=0
    backupdelta_part_x=0
    backupdelta_part_y=0

    def __init__(self,part_id,pre_part_x,pre_part_y):
        self.part_id = part_id
        self.pre_part_x= pre_part_x
        self.pre_part_y= pre_part_y
        self.updated= True
        self.backup_part_x=0
        self.backup_part_y=0
        self.delta_part_x=0
        self.delta_part_y=0
        self.backupdelta_part_x=0
        self.backupdelta_part_y=0

    def reset(self):
        self.updated = False

    def set_pre_part(self,pre_part_x,pre_part_y): # set bodypart position in the first frame 
        self.pre_part_x=pre_part_x
        self.pre_part_y=pre_part_y

    def set_backup(self,backup_part_x,backup_part_y): # set body part position each three frames
        self.backup_part_x =backup_part_x
        self.backup_part_y=backup_part_y

    def set_part_delta(self,delta_part_x,delta_part_y): # set bodypart detlta each frame
        self.delta_part_x=delta_part_x
        self.delta_part_y=delta_part_y
    def set_part_backupdelta(self,backupdelta_part_x,backupdelta_part_y):  # set bodypart detlta each  three frame
        self.backupdelta_part_x=backupdelta_part_x
        self.backupdelta_part_y=backupdelta_part_y
# person passing direction right curb or left curb
class FSM :
    number_of_states = 6 
    Current_state = 0
    passed_the_car = False
    def __init__(self,state):
        self.Current_state = state
        self.passed_the_car =False

    def get_output(self,new_direction):
        if new_direction == 0:
            if(self.Current_state==0):
                output=0
            else :
                self.Current_state-=1
                if self.Current_state <=2:
                    output =0
                else:
                    output =1
        else:
            if(self.Current_state ==5):
                output=1 
            else:
                self.Current_state =(self.Current_state +1) %self.number_of_states
                if self.Current_state >2 :
                    output=1
                else:
                    output=0
        return output


class Person:
    yaw=0
    passingtext =""
    headtext = ""
    pitch=0
    person_id = 0
    number_of_frames = 0
    person_last_frame =0
    person_update_frame=0
    prev_center =0 # pedestrian center at time t-5
    instant_prev_center =[0,0] # pedestrian center at time t-1
    prev_direction = 0 # pedestrian direction at time t-5 (left curb or right curb)
    pass_confidence =0
    direction_FSM = 0 
    pRear =0
    pLear=0
    pnose=0
    pLsholder=0
    pRsholder=0
    pRknee=0
    pLknee=0
    pRhip=0
    pLhip=0
    statex=0 #person  x position deltas in the paralellized image
    statey=0 #person y position deltas in the paralellized image
    lost_counter=0 # for how many frames the pedestrian was lost
    total_poss_conf = 0
    cont_pass_counter=0
    path=[]
    state =0 # 0=pass, 1=notpass
    CA_state =0 # notpassing 
    curbside = 0 # 0 --> left 1-->right
    added_flag =False
    cont_passing =False
    human_pos = False # true if the pedestrian body is detected
    person_bounding_box =[] 
    poss_array=[] # pixel position of the pedestrian 
    modified_poss_array=[] # parallelized pedestrian position
    pre_detected_parts=[] # bodyparts detected in the first frame
    detected_parts=[] # bodyparts detected 
    mov_direction = 0
    person_tier_average=[]
    person_tier_min=[]
    person_replace_old = False
    occulsion = False
    not_appearing = False
    lost = False
    recovery=0
    passing = False
    def __init__(self,person_bounding_box,person_id):
        self.person_bounding_box = person_bounding_box
        self.person_id = person_id
        self.mov_direction = ped_direction.directionest()
        self.yaw = 0
        self.pitch = 0
        self.recovery=0
        self.lost_counter=0
        self.passingtext=""
        self.headtext=""
        self.passing = False
        self.instant_prev_center =[0,0]
        self.number_of_frames  = 0 
        self.person_update_frame=0 
        self.state  = 0
        self.CA_state =0
        self.statex=0
        self.statey=0
        self.person_last_frame =0
        self.prev_center=0 
        self.direction_FSM = 0 
        self.prev_direction = 0 
        self.pass_confidence =0
        self.cont_pass_counter =0
        self.total_poss_conf = 0
        self.person_replace_old=False
        self.curbside  = 0
        self.added_flag  = False
        self.cont_passing =False
        self.human_pos = False
        self.occulsion = False
        self.not_appearing = False
        self.lost = False
        self.poss_array = []
        self.modified_poss_array=[]
        self.person_tier_average=[]
        self.person_tier_min=[]
        self.pRear =0
        self.pLear=0
        self.pnose=0
        self.pLsholder=0
        self.pRsholder=0
        self.pRknee=0
        self.pLknee=0
        self.pRhip=0
        self.pLhip=0
        self.detected_parts=[]
        self.pre_detected_parts=[]
        self.path=[]

    def getcenter(self): # get person boundingbox
        x = self.person_bounding_box[0]
        y = self.person_bounding_box[1]
        w = self.person_bounding_box[2]
        h = self.person_bounding_box[3]
        return [(x+(w/2)),(y+(h/2))]
           
def get_person_by_id(persons_array,p_id):
    for index,person in enumerate(persons_array) :
        if(person.person_id == p_id):
            return person,index

def same_person(person1 , persons_array):
    deltx = 10000000000000
    delty = 10000000000000
    deltah = 720 
    deltaw = 1280
    save_id =0
    x1,y1 = person1.getcenter()
    for person in persons_array: 
        poss = person.poss_array[-1]
        x2 = poss[0]
        y2 = poss[1] 
        if person.lost == False :
            dx = np.abs(x1-x2)
            dy = np.abs(y1-y2)
            dist = math.sqrt(dx**2 + dy**2)
            deltdist = math.sqrt(deltx**2 +delty**2)
            output.write("this person  delta dist : ")
            output.write(str(dist))
            person_id = person.person_id.urn
            person_id = person_id[9:]
            output.write(" id: ")
            output.write(person_id)
            output.write("\n")
            if(dist<deltdist):
                deltx = dx 
                delty = dy
                save_id = person.person_id
        else:
           hp1 = person1.person_bounding_box[3]
           wp1 = person1.person_bounding_box[2]
           hp = person.person_bounding_box[3]
           wp = person.person_bounding_box[2]
           deltahieght = abs(hp1-hp)
           deltawidth = abs(wp1-wp) 
           delta_y = abs(y1-y2)
           if deltahieght<deltah and deltawidth <deltaw and delta_y<delty :
               deltah = deltahieght
               deltaw = deltawidth
               deltax=0
               deltay=delta_y
               save_id=person.person_id


    return deltx ,delty ,save_id


counter = 0
faulty_predictions = 0
total_predictions=0
Rsholderconf= 0.15
Rearconf=0.10
Rhipconf = 0.10
Rkneeconf=0.25
Lkneeconf =0.25
noseconf =0.15
persons_array=[]
frame_persons=[]
Frame_head_posses=[]
x_change = open("x_change.txt","a") 
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
#delete
neede_id=0

def model_conf(state,prediction):
    global faulty_predictions
    if(state != prediction[0]):
        faulty_predictions+=1
    global total_predictions 
    total_predictions +=1


def remove_person_10_frames(current_frame):
    for person in persons_array :
        if(current_frame - person.person_update_frame >=10):
            persons_array.remove(person)
            output.write("removing______________________________________________________________________-----")
            output.write(str((person.person_id.urn)[9:]))
            output.write("\n")
    
def get_curb_side(person , car_pos):
    x_p,y_p = person.getcenter()

    if(person.number_of_frames==1):
        if x_p < car_pos[0]:
            person.direction_FSM = FSM(0)
            return 0 
        else :
            person.direction_FSM = FSM(5)
            return 1
            
    else:

            output.write("x_p :")
            output.write(str(x_p))
            output.write("prev center :")
            output.write(str(person.prev_center))
            output.write("\n")
            if x_p < car_pos[0] : # person on the left of the car 
                delta_prev = car_pos[0]-person.prev_center[0] 
                delta_current = car_pos[0]-x_p
                if(delta_prev > delta_current):
                    return person.direction_FSM.get_output(0)
                elif(delta_prev < delta_current):
                    if person.prev_direction == 0:
                        return person.direction_FSM.get_output(0)
                    else:
                        return person.direction_FSM.get_output(1)
                else:
                    return  person.direction_FSM.get_output(person.prev_direction)
            else: # person on the right of the car
                delta_prev = person.prev_center[0]-car_pos[0]
                delta_current = x_p-car_pos[0]
                if(delta_prev > delta_current):
                    return person.direction_FSM.get_output(1)
                elif(delta_prev < delta_current):
                    if person.prev_direction == 1:
                        return person.direction_FSM.get_output(1)
                    else:
                        return person.direction_FSM.get_output(0)
                else:
                    return  person.direction_FSM.get_output(person.prev_direction)
def pass_the_car(person):
    x_p,y_p = person.getcenter()
    if person.curbside == 0 and x_p > car_pos[0]+100 and person.direction_FSM.passed_the_car == False:
                person.direction_FSM.passed_the_car = True
    if person.curbside == 1 and x_p < car_pos[0]-100 and person.direction_FSM.passed_the_car == False:
                person.direction_FSM.passed_the_car = True
    output.write("\n passed the car or not--------- :")
    output.write(str(person.direction_FSM.passed_the_car))

def create_pre_unfound_parts(person):
    output.write("detected parts :")
    output.write(str(person.pre_detected_parts))
    output.write("\n")
    for part in range(0,18):
        if not(part in person.pre_detected_parts):
            if part ==0 : 
                person.pnose = Bodypart(0,0,0)
            elif part ==5 :
                person.pLsholder =Bodypart(5,0,0)
            elif part == 2 :
                person.pRsholder = Bodypart(2,0,0)
            elif part == 16 :
                person.pRear = Bodypart(16,0,0)
            elif part ==17 :
                person.pLear= Bodypart(17,0,0)
            elif part == 9 :
                person.pRknee = Bodypart(9,0,0)
            elif part == 12 :
                person.pLknee= Bodypart(12,0,0)
            elif part == 8 :
                person.pRhip = Bodypart(8,0,0)
            elif part ==11 :
                person.pLhip = Bodypart(11,0,0)
            
def total_pos_confedence(detected_parts,curbside): # need change
    total_cofidence = 0
    for part in detected_parts:
        if part == 0:
            total_cofidence +=noseconf
        elif (part == 2 and curbside ==0 ) or (part == 5 and curbside==1) :
            total_cofidence+=Rsholderconf
        elif part == 12 or part == 9:
            total_cofidence+=Rkneeconf
        elif (part == 11 and curbside ==1) or (part == 8 and curbside ==0) :
            total_cofidence+= Rhipconf
        elif (part ==16 and curbside ==0)or (part ==17 and curbside ==1):
            total_cofidence+= Rearconf
    return total_cofidence

 

def reset_added_flag():
    for person in persons_array :
        person.added_flag =False

def check_occ_or_not_appearing(person):
    box_x = person.person_bounding_box[0]
    box_y = person.person_bounding_box[1]
    box_w = person.person_bounding_box[2]
    box_h = person.person_bounding_box[3]
    persson_occ = None
    if box_w + box_x >= 1274 or box_x <=4:
        person.not_appearing = True
    if len(persons_array)>1:
        for person1 in persons_array :
            if (person.person_id)!= (person1.person_id):
                pbox_x = person1.person_bounding_box[0]
                pbox_y = person1.person_bounding_box[1]
                pbox_w = person1.person_bounding_box[2]
                pbox_h = person1.person_bounding_box[3]
                if (box_x in range(pbox_x,pbox_x+pbox_w+1)) or (box_x+box_w in range(pbox_x,pbox_x+pbox_w+1)):
                    person.occulsion=True
                    persson_occ = person1
    return persson_occ

def round_int(val):
    return int(round(val))






def lost_persons_edit():
    for person in persons_array:
       if person not in frame_persons:
           person.lost = True
           person.lost_counter+=1
           if person.lost_counter>=7:
                person.recovery=4

    persons_array.sort(key = lambda person: person.lost)


         
def match_head_poses(person,Frame_head_posses):
    x,y,w,h = person.person_bounding_box
    for head in Frame_head_posses:
        output.write("\n")
        output.write("heads : ")
        output.write(str(head))
        output.write("\n")
        head_x_min = head[0].item()
        head_y_min = head[1].item()
        head_x_max = head[2].item()
        head_y_max = head[3].item()
        if head_x_min >= x and head_x_min <= x+w and head_x_max >= x and head_x_max <= x+w   and head_y_min >=y and head_y_min <= y+h and head_y_max >=y and head_y_max <=y+h :
            output.write("--------")
            output.write("done")
            output.write("--------")
            person.yaw= head[4].item()
            person.pitch = head[5].item()
            output.write("\n")
            output.write("yaw : ")
            output.write(str(person.yaw))
            output.write(" pitch: ")
            output.write(str(person.pitch))
            output.write("\n")

def passing_confidence(person , ca_conf):
    if person.mov_direction.X_FSM.Current_state_x !=0:
        if person.mov_direction.X_FSM.Current_state_x >= 4 and person.mov_direction.X_FSM.prev_state_x <= person.mov_direction.X_FSM.Current_state_x:
            state_actual_conf = 0
        elif person.mov_direction.X_FSM.Current_state_x >= 4 and person.mov_direction.X_FSM.prev_state_x > person.mov_direction.X_FSM.Current_state_x:
            state_actual_conf =0.25
        else:
            state_actual_conf = ((person.mov_direction.X_FSM.Current_state_x *(50/3))+50)/100
         
    else:
        state_actual_conf = 0

    if person.human_pos and  ((person.total_poss_conf!=0 and(person.pass_confidence/person.total_poss_conf) >= 0.5) or person.pass_confidence==1 ) and  person.statex ==1 :
        actual_pose_conf = person.pass_confidence/person.total_poss_conf


        return (actual_pose_conf*0.5)+(state_actual_conf*0.5)

    elif person.CA_state ==1 and person.statex ==1:


        return (state_actual_conf*0.5) + (0.5*(person.CA_state))
    elif person.CA_state==1 and ca_conf==1 and person.mov_direction.X_FSM.Current_state_x <=4 and person.human_pos==False:

 
            return (state_actual_conf*0.5) + 0.5

    elif (person.mov_direction.X_FSM.Current_state_x <= 4) and person.CA_state==1 and ca_conf==1 and person.human_pos and(((person.total_poss_conf!=0 and(person.pass_confidence/person.total_poss_conf) >= 0.5) or person.pass_confidence ==1) ):
        actual_pose_conf = person.pass_confidence/person.total_poss_conf
  

        return (actual_pose_conf*1/3)+(state_actual_conf*1/3)+(1/3)
    elif person.statex==1 and person.CA_state==1 and person.human_pos and(((person.total_poss_conf!=0 and(person.pass_confidence/person.total_poss_conf) >= 0.5) or person.pass_confidence==1)):
        actual_pose_conf = person.pass_confidence/person.total_poss_conf


        return (actual_pose_conf*1/3)+(state_actual_conf*1/3)+((1/3))
    else:
        if person.human_pos ==False:


            return (state_actual_conf *0.5) + (person.CA_state * 0.5)
        else:
            if person.total_poss_conf !=0:
                actual_pose_conf = person.pass_confidence/person.total_poss_conf
            else:
                actual_pose_conf=0

            return (state_actual_conf * (1/3)) + (actual_pose_conf*(1/3))+ (person.CA_state *(1/3))


def visualize(frame_persons,image):
    for person in frame_persons:
        bounding = person.person_bounding_box
        x = bounding[0]
        y = bounding[1]
        w = bounding[2]
        h = bounding[3]
        person_id = person.person_id
        font_scale=1
        thickness=2
        if person.passing:
            color =(0,0,255)
        else:
            color = (51,255,51) 
        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
        cv2.putText(image, "state:"+person.passingtext, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale, color=color, thickness=thickness)
        cv2.putText(image, "id:"+str(person_id), (int(x), int(y) - 40), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale, color=color, thickness=thickness)
        if person.passing:
            image = draw.draw_path(image,person.path,person.person_bounding_box)

    cv2.imwrite("./needed_images/"+str(counter)+".jpg",image)   


def remove_disappearing_person(frame_persons):
    for person in frame_persons:
        x,y = person.getcenter()
        if person.passing:
            if person.curbside == 1:
                if x <=60:
                    persons_array.remove(person)
            else:
                if x >=1230:
                    persons_array.remove(person)
def search_for_person(person):
    global persons_array
    needed_person = None
    for person_1 in persons_array:
        if person_1.person_id == person.person_id :
            needed_person = person_1
    return needed_person
        
def draw_boxes(img, bbox,offset=(0,0)):
    for box in bbox:
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        # box text and bar   
        color = (51,255,51)
        cv2.rectangle(img, (x, y),(x+w,y+h), color, 2)
    return img

cam = cv2.VideoCapture('./images/H_passing_alone.mp4')
if not cam.isOpened():
    print("Unable to connect to camera.")
    exit(-1)
total_start = time.time()
while  cam.isOpened():
    
    ret,img = cam.read()
    if(ret==False):
        break
    img = cv2.resize(img,(1280,720))
    visualize_image = img.copy()
    Frame_head_posses = []
    #Frame_head_posses = Head.head_pose(img,counter)
    clear_img = img
    car_pos = (img.shape[1]/2,img.shape[0]/2) 
    print("car pos :",car_pos)
    start_time = time.time()
    outputs,boundingbox = tu.detect(img)
    img = draw_boxes(img,outputs)
    #cropped_images,boundingbox_dim,img = yt.get_box(img)
    yolo_end =time.time()
    number_of_persons = len(outputs)
    frame_persons=[]
    print(number_of_persons)
    output.write("\n")
    output.write("counter = ")
    output.write(str(counter))
    output.write("\n")
    print("yolo taken time :",yolo_end-start_time)

    for output in outputs:
        training_samples =[]
        dim = output[:4]
        print(str(dim))
        person = Person(dim,output[4])
        person.poss_array.append(person.getcenter())
       
        if(counter ==0 or len(persons_array)==0):
            persons_array.append(person)
            person.number_of_frames+=1
            person.added_flag = True
            neede_id = person.person_id
        else:
            old_person = search_for_person(person)
            if old_person == None:
                persons_array.append(person)
                person.number_of_frames+=1
                person.added_flag = True
                neede_id = person.person_id
            else:

                old_person.number_of_frames+=1
                old_person.poss_array = old_person.poss_array + person.poss_array
                old_person.modified_poss_array = old_person.modified_poss_array + person.modified_poss_array
                person = old_person
                print("person pose array length:",len(person.poss_array))
                person.person_bounding_box = dim
                person.added_flag=True
                person.person_replace_old =True
                person.human_pos = False
        if person.recovery>0:
            person.recovery-=1
            output.write("\n recovery :")
            output.write(str(person.recovery))
            output.write("\n")
        person.lost_counter=0

        frame_persons.append(person)
        center = person.getcenter()
        center_x = center[0]
        center_y = center[1]
        output.write("center_x : ")
        output.write(str(center_x))
        output.write("   center_y : ")
        output.write(str(center_y))
        output.write("\n")
        if(person.number_of_frames == 1):
            person.person_update_frame = counter
        else:
            if(counter - person.person_update_frame > 1):
                print("this person where lost")
                person_id = person.person_id
                output.write("id")
                output.write(person_id)
                output.write("this person was lost")
                output.write("\n")
                person.person_update_frame =counter
            else:
                person.person_update_frame =counter

        if((person.number_of_frames-1) % 5 == 0):
            if person.number_of_frames >1:
                person.direction_FSM.passed_the_car=False
            direction = get_curb_side(person,car_pos)
            person.curbside = direction
            output.write("this person direction is :  ")
            output.write(str(direction))
            output.write("    the FSM STATE is  :  ")
            output.write(str(person.direction_FSM.Current_state))
            output.write("    prev center is  :  ")
            output.write(str(person.prev_center))
            output.write("   passed the car  :  ")
            output.write(str(person.direction_FSM.passed_the_car))
            output.write("\n")
            person.prev_center=person.getcenter()
            person.prev_direction = direction
        pass_the_car(person)
        image_h, image_w = img.shape[:2]
        img_1 = np.zeros([image_h,image_w,3],dtype=np.uint8)
        img_1.fill(255)
        ytx = dim[0]
        yty = dim[1]
        w= dim[2]
        h=dim[3]
        img_1[yty-7:yty+7+h,ytx-7:ytx+w+7] = img[yty-7:yty+7+h,ytx-7:ytx+w+7]
        #match_head_poses(person,Frame_head_posses)
        if(img_1.shape[0]>1000):
            resized_image = cv2.resize(img_1,(640,368))
            cv2.imshow("image",resized_image)
            cv2.waitKey(1) 
        else:    
            cv2.imshow("image",img_1)
            cv2.waitKey(1)   
        if person.number_of_frames ==1:
            person.instant_prev_center[0],_=td.get_deltas(img_1,counter) 
            _,person.instant_prev_center[1]=person.getcenter() 
            person.modified_poss_array.append(person.instant_prev_center)
        else:
            center_x,_= td.get_deltas(img_1,counter)
            _,center_y= person.getcenter() 
            output.write("\n")
            output.write("hologram center x :")
            output.write(str(center_x))
            output.write("\n")
            if person.curbside==0: #left-->right
                if center_y >288 and center_y<= 360 :
                    person.person_tier_average=left_curb_tier5
                    person.person_tier_min=min_left_curb_tier5
                elif center_y>360 and center_y<=432:
                    person.person_tier_average=left_curb_tier6
                    person.person_tier_min=min_left_curb_tier6
                else:
                    person.person_tier_average=left_curb_tier7
                    person.person_tier_min=min_left_curb_tier7
            else:
                if center_y>288 and center_y<= 360 :
                    person.person_tier_average=right_curb_tier5
                    person.person_tier_min=min_right_curb_tier5
                elif center_y>360 and center_y<=432:
                    person.person_tier_average=right_curb_tier6
                    person.person_tier_min=min_right_curb_tier6
                else:
                    person.person_tier_average=right_curb_tier7
                    person.person_tier_min=min_right_curb_tier7

            person.modified_poss_array.append([center_x,center_y])
            del_x = center_x - person.instant_prev_center[0]
            del_y = center_y - person.instant_prev_center[1]
            ncx,ncy = person.getcenter() 
            occ_persons = check_occ_or_not_appearing(person)
            person.statex,person.statey = person.mov_direction.get_direction([ncx,ncy],del_x,del_y,person.curbside,person.not_appearing,person.occulsion)
            if person.person_id == neede_id:
                if person.occulsion :
                    x_change.write("\n")
                    x_change.write("occulsionnn")
                    x_change.write("\n")
                if person.not_appearing:
                    x_change.write("\n")
                    x_change.write("not apperainggg")
                    x_change.write("\n")
                x_change.write("x = ")
                x_change.write(str(center_x))
                x_change.write("   y =  ")
                x_change.write(str(center_y))
                x_change.write("   counter = ")
                x_change.write(str(counter))
                x_change.write("\n")
                x_change.write("delta x = ")
                x_change.write(str(del_x))
                x_change.write("  delta y = ")
                x_change.write(str(del_y))
                x_change.write("\n")
                x_change.write("fsm state x = ")
                x_change.write(str(person.mov_direction.X_FSM.Current_state_x))
                x_change.write("\n")
                x_change.write("statex = ")
                x_change.write(str(person.statex))
                x_change.write("  statey = ")
                x_change.write(str(person.statey))
                # if person.statex ==1:
                #     cv2.imwrite("./needed_images/needed"+str(counter)+"frames"+str(person.number_of_frames)+".jpg",img)
                x_change.write("\n")
                x_change.write("-----------------------------")
                x_change.write("\n")
            person.instant_prev_center = [center_x,center_y]        
        humans=[]        
        humans = wc.getpose(img_1)
        if(len(humans)!=0):
            humans = [humans[0]] 
            for i,human in enumerate(humans) :
                person.human_pos = True
                output.write('\n')
                output.write("body : ")
                output.write(str(human))
                output.write('\n')
                for bodypart in human.body_parts:
                       
                    if(person.number_of_frames==1):
                        if(bodypart == 0 and (human.body_parts[bodypart].score)>0.5): #nose 
                            person.pnose=Bodypart(0,round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                            person.pnose.set_backup(round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                            person.pre_detected_parts.append(0)
                        elif(bodypart==5 and (human.body_parts[bodypart].score)>0.5):
                            person.pLsholder=Bodypart(5,round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                            person.pLsholder.set_backup(round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                            person.pre_detected_parts.append(5)
                        elif(bodypart==2 and (human.body_parts[bodypart].score)>0.5):
                            person.pRsholder=Bodypart(2,round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                            person.pRsholder.set_backup(round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                            person.pre_detected_parts.append(2)
                        elif(bodypart==16 and (human.body_parts[bodypart].score)>0.5):
                            person.pRear=Bodypart(16,round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                            person.pRear.set_backup(round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                            person.pre_detected_parts.append(16)
                        elif(bodypart==17 and (human.body_parts[bodypart].score)>0.5):
                            person.pLear=Bodypart(17,round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                            person.pLear.set_backup(round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                            person.pre_detected_parts.append(17)
                        elif(bodypart==9):#right knee
                            person.pRknee=Bodypart(9,round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                            person.pRknee.set_backup(round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                            person.pre_detected_parts.append(9)
                        elif(bodypart==12 and (human.body_parts[bodypart].score)>0.5):
                            person.pLknee=Bodypart(12,round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                            person.pLknee.set_backup(round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                            person.pre_detected_parts.append(12)
                        elif bodypart==8 and (human.body_parts[bodypart].score)>0.5:
                            person.pRhip=Bodypart(8,round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                            person.pRhip.set_backup(round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                            person.pre_detected_parts.append(8)
                        elif bodypart==11 and (human.body_parts[bodypart].score)>0.5:
                            person.pLhip=Bodypart(11,round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                            person.pLhip.set_backup(round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                            person.pre_detected_parts.append(11)
                        create_pre_unfound_parts(person)                           
                    else:
                        if(bodypart == 0 and (human.body_parts[bodypart].score)>0.5): #nose 
                            Nosex =round_int((human.body_parts[bodypart].x)*image_w)
                            Nosey =round_int((human.body_parts[bodypart].y)*image_h)
                            person.detected_parts.append(0)
                            if(person.pnose ==0 or  not(bodypart in person.pre_detected_parts)):
                                person.pnose=Bodypart(0,round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                                person.pnose.set_backup(round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                                person.pnose.reset()
                            else:
                                deltanosex = Nosex-person.pnose.pre_part_x
                                deltanosey = Nosey-person.pnose.pre_part_y
                                person.pnose.set_part_delta(deltanosex,deltanosey)
                                person.pnose.set_pre_part(Nosex,Nosey)
                            
                                if(person.number_of_frames%3==0):
                                    backupdeltanosex = Nosex - person.pnose.backup_part_x
                                    backupdeltanosey = Nosey - person.pnose.backup_part_y
                                    person.pnose.set_part_backupdelta(backupdeltanosex,backupdeltanosey)
                                    person.pnose.set_backup(Nosex,Nosey)

                        elif(bodypart==5 and (human.body_parts[bodypart].score)>0.5):
                            Lsholderx=round_int((human.body_parts[bodypart].x)*image_w)
                            Lsholdery =round_int((human.body_parts[bodypart].y)*image_h)
                            person.detected_parts.append(5)
                            if(person.pLsholder ==0 or  not(bodypart in person.pre_detected_parts)):
                                person.pLsholder=Bodypart(5,round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                                person.pLsholder.set_backup(round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                                person.pLsholder.reset()
                            else:
                                deltaLsholderx = Lsholderx-person.pLsholder.pre_part_x
                                deltaLsholdery = Lsholdery-person.pLsholder.pre_part_y
                                person.pLsholder.set_part_delta(deltaLsholderx,deltaLsholdery)
                                person.pLsholder.set_pre_part(Lsholderx,Lsholdery)
                                if(person.number_of_frames%3==0):
                                    backupdeltaLsholderx = Lsholderx - person.pLsholder.backup_part_x
                                    backupdeltaLsholdery = Lsholdery - person.pLsholder.backup_part_y
                                    person.pLsholder.set_part_backupdelta(backupdeltaLsholderx,backupdeltaLsholdery)
                                    person.pLsholder.set_backup(Lsholderx,Lsholdery)

                        elif(bodypart==2 and (human.body_parts[bodypart].score)>0.5):
                            Rsholderx=round_int((human.body_parts[bodypart].x)*image_w)
                            Rsholdery =round_int((human.body_parts[bodypart].y)*image_h)
                            person.detected_parts.append(2)
                            if(person.pRsholder ==0 or  not(bodypart in person.pre_detected_parts)):
                                person.pRsholder=Bodypart(2,round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                                person.pRsholder.set_backup(round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                                person.pRsholder.reset()
                            else:
                                deltaRsholderx = Rsholderx-person.pRsholder.pre_part_x
                                deltaRsholdery = Rsholdery-person.pRsholder.pre_part_y
                                person.pRsholder.set_part_delta(deltaRsholderx,deltaRsholdery)
                                person.pRsholder.set_pre_part(Rsholderx,Rsholdery)
                                if(person.number_of_frames%3==0):
                                    backupdeltaRsholderx = Rsholderx - person.pRsholder.backup_part_x
                                    backupdeltaRsholdery = Rsholdery - person.pRsholder.backup_part_y
                                    person.pRsholder.set_part_backupdelta(backupdeltaRsholderx,backupdeltaRsholdery)
                                    person.pRsholder.set_backup(Rsholderx,Rsholdery)

                        elif(bodypart==16 and (human.body_parts[bodypart].score)>0.5):
                            Rearx=round_int((human.body_parts[bodypart].x)*image_w)
                            Reary =round_int((human.body_parts[bodypart].y)*image_h)
                            person.detected_parts.append(16)
                            if(person.pRear ==0 or  not(bodypart in person.pre_detected_parts)):
                                person.pRear=Bodypart(16,round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                                person.pRear.set_backup(round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                                person.pRear.reset()
                            else:
                                deltaRearx = Rearx-person.pRear.pre_part_x
                                deltaReary = Reary-person.pRear.pre_part_y
                                person.pRear.set_part_delta(deltaRearx,deltaReary)
                                person.pRear.set_pre_part(Rearx,Reary)
                                if(person.number_of_frames%3==0):
                                    backupdeltaRearx = Rearx - person.pRear.backup_part_x
                                    backupdeltaReary = Reary - person.pRear.backup_part_y
                                    person.pRear.set_part_backupdelta(backupdeltaRearx,backupdeltaReary)
                                    person.pRear.set_backup(Rearx,Reary)

                        elif(bodypart==17 and (human.body_parts[bodypart].score)>0.5):
                            Learx=round_int((human.body_parts[bodypart].x)*image_w)
                            Leary =round_int((human.body_parts[bodypart].y)*image_h)
                            person.detected_parts.append(17)
                            if(person.pLear ==0 or  not(bodypart in person.pre_detected_parts)):
                                person.pLear=Bodypart(17,round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                                person.pLear.set_backup(round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                                person.pLear.reset()
                            else:
                                deltaLearx = Learx-person.pLear.pre_part_x
                                deltaLeary = Leary-person.pLear.pre_part_y
                                person.pLear.set_part_delta(deltaLearx,deltaLeary)
                                person.pLear.set_pre_part(Learx,Leary)
                                if(person.number_of_frames%3==0):
                                    backupdeltaLearx = Learx - person.pLear.backup_part_x
                                    backupdeltaLeary = Leary - person.pLear.backup_part_y
                                    person.pLear.set_part_backupdelta(backupdeltaLearx,backupdeltaLeary)
                                    person.pLear.set_backup(Learx,Leary)
                        elif(bodypart==9 and (human.body_parts[bodypart].score)>0.5):#right knee
                            Rknee=round_int((human.body_parts[bodypart].x)*image_w)
                            Rkneey =round_int((human.body_parts[bodypart].y)*image_h)
                            person.detected_parts.append(9)
                            if(person.pRknee ==0 or  not(bodypart in person.pre_detected_parts)):
                                person.pRknee=Bodypart(9,round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                                person.pRknee.set_backup(round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                                person.pRknee.reset()
                            else:
                                deltaRknee = Rknee-person.pRknee.pre_part_x
                                deltaRkneey = Rkneey-person.pRknee.pre_part_y
                                person.pRknee.set_part_delta(deltaRknee,deltaRkneey)
                                person.pRknee.set_pre_part(Rknee,Rkneey)
                                if(person.number_of_frames%3==0):
                                    backupdeltaRknee = Rknee - person.pRknee.backup_part_x
                                    backupdeltaRkneey = Rkneey - person.pRknee.backup_part_y
                                    person.pRknee.set_part_backupdelta(backupdeltaRknee,backupdeltaRkneey)
                                    person.pRknee.set_backup(Rknee,Rkneey)
                        elif(bodypart==12 and (human.body_parts[bodypart].score)>0.5):
                            Lknee=round_int((human.body_parts[bodypart].x)*image_w)
                            Lkneey =round_int((human.body_parts[bodypart].y)*image_h)
                            person.detected_parts.append(12)
                            if(person.pLknee ==0 or  not(bodypart in person.pre_detected_parts)):
                                person.pLknee=Bodypart(12,round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                                person.pLknee.set_backup(round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                                person.pLknee.reset()
                            else:
                                deltaLknee = Lknee-person.pLknee.pre_part_x
                                deltaLkneey = Lkneey-person.pLknee.pre_part_y
                                person.pLknee.set_part_delta(deltaLknee,deltaLkneey)
                                person.pLknee.set_pre_part(Lknee,Lkneey)
                                if(person.number_of_frames%3==0):
                                    backupdeltaLknee = Lknee - person.pLknee.backup_part_x
                                    backupdeltaLkneey = Lkneey - person.pLknee.backup_part_y
                                    person.pLknee.set_part_backupdelta(backupdeltaLknee,backupdeltaLkneey)
                                    person.pLknee.set_backup(Lknee,Lkneey)
                            
                        elif bodypart==8 and (human.body_parts[bodypart].score)>0.5:
                            Rhip=round_int((human.body_parts[bodypart].x)*image_w)
                            Rhipy =round_int((human.body_parts[bodypart].y)*image_h)
                            person.detected_parts.append(8)
                            if(person.pRhip ==0 or  not(bodypart in person.pre_detected_parts)):
                                person.pRhip=Bodypart(8,round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                                person.pRhip.set_backup(round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                                person.pRhip.reset()
                            else:
                                deltaRhip = Rhip-person.pRhip.pre_part_x
                                deltaRhipy = Rhipy-person.pRhip.pre_part_y
                                person.pRhip.set_part_delta(deltaRhip,deltaRhipy)
                                person.pRhip.set_pre_part(Rhip,Rhipy)

                        elif bodypart==11 and (human.body_parts[bodypart].score)>0.5:
                            Lhip=round_int((human.body_parts[bodypart].x)*image_w)
                            Lhipy =round_int((human.body_parts[bodypart].y)*image_h)
                            person.detected_parts.append(11)
                            if(person.pLhip ==0 or  not(bodypart in person.pre_detected_parts)):
                                person.pLhip=Bodypart(11,round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                                person.pLhip.set_backup(round_int((human.body_parts[bodypart].x)*image_w),round_int((human.body_parts[bodypart].y)*image_h))
                                person.pLhip.reset()
                            else:
                                deltaLhip = Lhip-person.pLhip.pre_part_x
                                deltaLhipy = Lhipy-person.pLhip.pre_part_y
                                person.pLhip.set_part_delta(deltaLhip,deltaLhipy)
                                person.pLhip.set_pre_part(Lhip,Lhipy)
                        if person.person_replace_old :
                            persons_array[index]=person
                            person.person_replace_old=False
                if person.number_of_frames >1:
                    person.pre_detected_parts =person.detected_parts
                    create_pre_unfound_parts(person)

                passRear=0
                passRsholder=0
                passnose =0
                passRhip =0
                passRknee=0
                passLknee=0
                passLear=0
                passLsholder=0
                passLhip=0
                Rkneedistance=0
                nosedistance=0
                Rsholderdistance=0
                Lkneedistance=0
                Reardistance=0
                Lsholderdistance =0
                Leardistance=0
                print("frames for each person :",person.number_of_frames)
                
                if(person.number_of_frames>1):
                    if(person.curbside):
                        output.write("this person is passing from the RIGHT CURB")
                        output.write("\n")
                        if person.pRknee.delta_part_x <0:
                            Rkneedistance =np.sqrt(person.pRknee.delta_part_x**2 + person.pRknee.delta_part_y**2)
                        if person.pnose.delta_part_x<0:
                            nosedistance= np.sqrt(person.pnose.delta_part_x**2 + person.pnose.delta_part_y**2)
                        if person.pLsholder.delta_part_x<0:
                            Lsholderdistance = np.sqrt(person.pLsholder.delta_part_x**2 + person.pLsholder.delta_part_y**2)
                        if person.pLear.delta_part_x <0:
                            Leardistance = np.sqrt(person.pLear.delta_part_x**2 + person.pLear.delta_part_x**2)
                        if person.pLknee.delta_part_x<0:
                            Lkneedistance = np.sqrt(person.pLknee.delta_part_x**2 + person.pLknee.delta_part_x**2)
                        
                        # if((-1)*(person.pLear.delta_part_x)>person.person_tier_min[2] and (-1)*(person.pLear.backupdelta_part_x)>person.person_tier_min[16]): #or Leardistance >=2):
                        #     passLear=0.75
                        # if((-1)*(person.pLknee.delta_part_x)>person.person_tier_min[12] and (-1)*(person.pLknee.backupdelta_part_x) >person.person_tier_min[21]):# or Lkneedistance>=2 ):
                        #     passLknee=0.75
                        # if((-1)*(person.pLsholder.delta_part_x)>person.person_tier_min[6] and (-1)*(person.pLsholder.backupdelta_part_x)>person.person_tier_min[18]):# or Lsholderdistance>=2):
                        #     passLsholder=0.75
                        # if((-1)*(person.pnose.delta_part_x)>person.person_tier_min[5] and (-1)*(person.pnose.backupdelta_part_x)>person.person_tier_min[17]):# or nosedistance>=2):
                        #     passnose=0.75
                        # if((-1)*(person.pRknee.delta_part_x)>person.person_tier_min[10] and (-1)*(person.pRknee.backupdelta_part_x)>person.person_tier_min[20]):# or Rkneedistance >=2):
                        #     passRknee=0.75
                        # if((-1)*(person.pLhip.delta_part_x)>person.person_tier_min[13]):# this have a problem
                        #     passLhip=0.75

                        if((-1)*(person.pLear.delta_part_x)>=person.person_tier_average[2] or (-1)*(person.pLear.backupdelta_part_x)>person.person_tier_average[16]): #or Leardistance >=2):
                            passLear=1
                        elif((-1)*(person.pLear.delta_part_x)>=person.person_tier_min[2]):# and (-1)*(person.pLear.backupdelta_part_x)>person.person_tier_min[16]): #or Leardistance >=2):
                            passLear=0.35
                        if((-1)*(person.pLknee.delta_part_x)>=person.person_tier_average[12] or (-1)*(person.pLknee.backupdelta_part_x) >person.person_tier_average[21]):# or Lkneedistance>=2 ):
                            passLknee=1
                        elif((-1)*(person.pLknee.delta_part_x)>=person.person_tier_min[12]):# and (-1)*(person.pLknee.backupdelta_part_x) >person.person_tier_min[21]):# or Lkneedistance>=2 ):
                            passLknee=0.65
                         
                        if((-1)*(person.pLsholder.delta_part_x)>=person.person_tier_average[6] or (-1)*(person.pLsholder.backupdelta_part_x)>person.person_tier_average[18]):# or Lsholderdistance>=2):
                            passLsholder=1
                        elif((-1)*(person.pLsholder.delta_part_x)>=person.person_tier_min[6]):# and (-1)*(person.pLsholder.backupdelta_part_x)>person.person_tier_min[18]):# or Lsholderdistance>=2):
                            passLsholder=0.65
                        if((-1)*(person.pnose.delta_part_x)>=person.person_tier_average[5] or (-1)*(person.pnose.backupdelta_part_x)>person.person_tier_average[17]):# or nosedistance>=2):
                            passnose=1
                        elif((-1)*(person.pnose.delta_part_x)>=person.person_tier_min[5]):# and (-1)*(person.pnose.backupdelta_part_x)>person.person_tier_min[17]):# or nosedistance>=2):
                            passnose=0.35
                        if((-1)*(person.pRknee.delta_part_x)>=person.person_tier_average[10] or (-1)*(person.pRknee.backupdelta_part_x)>person.person_tier_average[20]):# or Rkneedistance >=2):
                            passRknee=1
                        elif((-1)*(person.pRknee.delta_part_x)>=person.person_tier_min[10]):# and (-1)*(person.pRknee.backupdelta_part_x)>person.person_tier_min[20]):# or Rkneedistance >=2):
                            passRknee=0.65
                        if((-1)*(person.pLhip.delta_part_x)>=person.person_tier_average[13]):# this have a problem
                            passLhip=1
                        elif ((-1)*(person.pLhip.delta_part_x)>=person.person_tier_min[13]):# this have a problem
                            passLhip=0.35

                        person.pass_confidence = (passLear*Rearconf)+(passLsholder*Rsholderconf)+(passRknee*Rkneeconf)+(passnose*noseconf)+(Rhipconf*passRhip)+(Lkneeconf*passLknee)
                        
                        training_samples.append([(-1)*(person.pRear.delta_part_x),(-1)*(person.pRear.delta_part_y),(-1)*(person.pLear.delta_part_x),(-1)*(person.pLear.delta_part_y),
                        (-1)*(person.pnose.delta_part_y),(-1)*(person.pnose.delta_part_x),(-1)*(person.pLsholder.delta_part_x),(-1)*(person.pLsholder.delta_part_y),(-1)*(person.pRsholder.delta_part_x),
                        (-1)*(person.pRsholder.delta_part_y),(-1)*(person.pRknee.delta_part_x),(-1)*(person.pRknee.delta_part_y),(-1)*(person.pLknee.delta_part_x),
                        (-1)*(person.pRhip.delta_part_x),(-1)*(person.pRhip.delta_part_y),(-1)*(person.pRear.backupdelta_part_x),(-1)*(person.pLear.backupdelta_part_x),
                        (-1)*(person.pnose.backupdelta_part_x),(-1)*(person.pLsholder.backupdelta_part_x),(-1)*(person.pRsholder.backupdelta_part_x),
                        (-1)*(person.pRknee.backupdelta_part_x),(-1)*(person.pLknee.backupdelta_part_x)])
                    else:

                        if person.pRknee.delta_part_x >0:
                            Rkneedistance =np.sqrt(person.pRknee.delta_part_x**2 + person.pRknee.delta_part_y**2)
                        if person.pnose.delta_part_x > 0:
                            nosedistance= np.sqrt(person.pnose.delta_part_x**2 + person.pnose.delta_part_y**2)
                        if person.pRsholder.delta_part_x > 0:
                            Rsholderdistance = np.sqrt(person.pRsholder.delta_part_x**2 + person.pRsholder.delta_part_y**2)
                        if person.pRear.delta_part_x >0:
                            Reardistance = np.sqrt(person.pRear.delta_part_x**2 + person.pRear.delta_part_x**2)
                        if person.pLknee.delta_part_x > 0:
                            Lkneedistance = np.sqrt(person.pLknee.delta_part_x**2 + person.pLknee.delta_part_x**2)
                        if( person.pRear.backupdelta_part_x >=person.person_tier_average[15] or person.pRear.delta_part_x>=person.person_tier_average[0]):# or Reardistance >=2):
                            passRear=1
                        elif person.pRear.backupdelta_part_x >=person.person_tier_min[15]:
                            passRear=0.35
                        if( person.pRknee.backupdelta_part_x >= person.person_tier_average[20] or person.pRknee.delta_part_x>=person.person_tier_average[10]):# or Rkneedistance >=1):
                            passRknee=1
                        elif person.pRknee.backupdelta_part_x >= person.person_tier_min[20]:
                            passRknee =0.65
                        if( person.pRsholder.backupdelta_part_x>=person.person_tier_average[19] or person.pRsholder.delta_part_x>=person.person_tier_average[8]):# or Rsholderdistance >=2):
                            passRsholder=1
                        elif person.pRsholder.backupdelta_part_x >= person.person_tier_min[19]:
                            passRsholder=0.65
                        if( person.pnose.backupdelta_part_x >=person.person_tier_average[17] or person.pnose.delta_part_x>=person.person_tier_average[5]): #or nosedistance >=2):
                            passnose=1
                        elif person.pnose.backupdelta_part_x >= person.person_tier_min[17]:
                            passnose=0.35
                        if(person.pLknee.delta_part_x>=person.person_tier_average[12] ):#or person.pLknee.backupdelta_part_x >=person.person_tier_average[21]):# or Lkneedistance >= 1):
                            passLknee=1
                        elif person.pLknee.backupdelta_part_x >= person.person_tier_min[12]:
                            passLknee=0.65
                        if(person.pRhip.delta_part_x>=person.person_tier_average[13]):
                            passRhip=1
                        elif person.pRhip.backupdelta_part_x >= person.person_tier_min[13]:
                            passRhip=0.35
                        person.pass_confidence = (passRear*Rearconf)+(passRsholder*Rsholderconf)+(passRknee*Rkneeconf)+(passnose*noseconf)+(Rhipconf*passRhip)+(Lkneeconf*passLknee)
                        training_samples.append([(person.pRear.delta_part_x),(person.pRear.delta_part_y),(person.pLear.delta_part_x),(person.pLear.delta_part_y),
                        (person.pnose.delta_part_y),(person.pnose.delta_part_x),(person.pLsholder.delta_part_x),(person.pLsholder.delta_part_y),(person.pRsholder.delta_part_x),
                        (person.pRsholder.delta_part_y),(person.pRknee.delta_part_x),(person.pRknee.delta_part_y),(person.pLknee.delta_part_x),
                        (person.pRhip.delta_part_x),(person.pRhip.delta_part_y),(person.pRear.backupdelta_part_x),(person.pLear.backupdelta_part_x),
                        (person.pnose.backupdelta_part_x),(person.pLsholder.backupdelta_part_x),(person.pRsholder.backupdelta_part_x),
                        (person.pRknee.backupdelta_part_x),(person.pLknee.backupdelta_part_x)])
                else:
                    training_samples.append([(person.pRear.delta_part_x),(person.pRear.delta_part_y),(person.pLear.delta_part_x),(person.pLear.delta_part_y),
                    (person.pnose.delta_part_y),(person.pnose.delta_part_x),(person.pLsholder.delta_part_x),(person.pLsholder.delta_part_y),(person.pRsholder.delta_part_x),
                    (person.pRsholder.delta_part_y),(person.pRknee.delta_part_x),(person.pRknee.delta_part_y),(person.pLknee.delta_part_x),
                    (person.pRhip.delta_part_x),(person.pRhip.delta_part_y),(person.pRear.backupdelta_part_x),(person.pLear.backupdelta_part_x),
                    (person.pnose.backupdelta_part_x),(person.pLsholder.backupdelta_part_x),(person.pRsholder.backupdelta_part_x),
                    (person.pRknee.backupdelta_part_x),(person.pLknee.backupdelta_part_x)])
                    person.pass_confidence = 0

                if(person.number_of_frames>1 and(person.pRknee.delta_part_x > (person.person_tier_average[10])+2) and person.curbside ==0):
                    person.pass_confidence = 1.0
                elif(person.number_of_frames>1 and((-1)*person.pRknee.delta_part_x >= person.person_tier_average[10]+2 ) and person.curbside ==1):
                    person.pass_confidence = 1.0
                training_samples = np.array(training_samples)
                output.write("DELTAS: ")
                output.write(str(training_samples))
                output.write("\n")

                if(person.pass_confidence >= 0.5):
                    if(person.yaw ==0 and person.pitch ==0):
                        print("warning this person is not looking") 
                    else :
                        print("this person is looking yaw :",person.yaw) 
                        print("human number ",person.person_id,"is passing with percentage",int((person.pass_confidence*100)),"%")
                    person.state = 1
                else:
                    if(person.yaw ==0 and person.pitch ==0):
                        print("warning this person is not looking") 
                    else :
                        print("this person is looking yaw :",person.yaw) 
                    if((person.state ==1 or person.cont_passing) and person.pass_confidence > 0):
                        print("person : ",person.person_id," is cont passing")
                        person.cont_pass_counter+=1
                        person.state = 0
                        person.cont_passing =True
                        print(person.cont_pass_counter)
                        if(person.cont_pass_counter==3):
                            person.cont_passing =False
                            person.cont_pass_counter =0 
                    else:
                        print("person",person.person_id,"not passing")
                        person.state=0
                print("person bounding box = ",person.person_bounding_box)
                print("state :",person.state)
            if len(person.pre_detected_parts)==0:
                person.human_pos=False
            person.total_poss_conf = total_pos_confedence(person.pre_detected_parts,person.curbside)
            
            person.detected_parts=[]
            person_id = person.person_id
            output.write("id : ")
            output.write(person_id)
            output.write("\n")
            output.write("person bounding box :")
            output.write(str(person.person_bounding_box ))
            output.write("\n")
            output.write("personstate :")
            output.write(str(person.state ))
            output.write("\n")
            output.write("pass conf :")
            output.write(str(person.pass_confidence))
            output.write("\n")
            output.write(" total pass conf :")
            output.write(str(person.total_poss_conf))
            output.write("\n")
            output.write( "number of persons :")
            output.write(str(number_of_persons))

            
        else:
            training_samples =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            person.pre_detected_parts=[]
            person.pass_confidence = 0 
            output.write("no human")
            create_pre_unfound_parts(person)

        if person.number_of_frames > 1 :
            output.write("person pos array before :")
            output.write(str(person.modified_poss_array))
            output.write("\n")
            xpt,ypt,variance = CA.predict(person.modified_poss_array,person.recovery)
            x_values,_,_ = CA.predict(person.poss_array,person.recovery)
            person.path=x_values
            output.write("pixels poss :")
            output.write(str(x_values))
            output.write("\n")
            if variance >5 :
                ca_conf = 0.5
            else:
                ca_conf = 1
            if(person.number_of_frames >=4):
                last_position = (person.modified_poss_array[-1][0] +person.modified_poss_array[-2][0] +person.modified_poss_array[-3][0]) /3
            else:
                last_position = person.modified_poss_array[-1][0]
            print("last poss xt in the array is :",last_position ,"xpt[-1]",xpt[-1])
            if person.curbside :
                deltax = person.modified_poss_array[-1][0]-xpt[-1] 
            else:
                deltax = xpt[-1] -person.modified_poss_array[-1][0]
            output.write("deltax is =")
            output.write(str(deltax))
            if(deltax >= 14):
                person.CA_state = 1

                print("accordingly this person is still moving in the same direction")
                output.write("CA Predict passing")
            else:
                person.CA_state = 0
                print("this person is not moving")
                output.write("CA Predict NOT passing")
            output.write("\n")
            output.write("xpt[-1] :")
            output.write(str(xpt[-1]))
            output.write("   variance:")
            output.write(str(variance))
            

        else: 
            person.pass_confidence = 0
            print("sorry but person ",person.person_id," in unpredictable ")
        output.write("\n")
        if person.number_of_frames>1:
            person_cof = passing_confidence(person , ca_conf)
            output.write("\n")
            output.write("this person is passing with percentage :")
            output.write(str(person_cof*100))
            output.write("\n")
        print(training_samples)
        model_data = training_samples
        model_data=np.append(model_data,person.CA_state)
        model_data=np.append(model_data,person.statex)
        model_data=np.append(model_data,person.curbside)
        print(model_data.shape)
        model_data= np.reshape(model_data,(1,model_data.shape[0]))
        prediction = loaded_model.predict_classes(model_data)
        print("what the model think is :",prediction)
        output.write("model predicts :")
        output.write(str(prediction))
        output.write("\n")
        model_conf(person.state,prediction)
        #if person.person_id != neede_id:
            #if not(person.direction_FSM.passed_the_car):
        if person.human_pos and  ((person.total_poss_conf!=0 and(person.pass_confidence/person.total_poss_conf) >= 0.5) or person.pass_confidence==1 ) and  person.statex ==1 :
            person.passingtext ="Passing"
            person.passing=True
            

        elif person.CA_state ==1 and person.statex ==1:
            person.passingtext="Passing"
            person.passing=True

        elif person.CA_state==1  and person.mov_direction.X_FSM.Current_state_x <=4 and person.human_pos==False:
                person.passingtext="Passing"
                person.passing=True

        elif (person.mov_direction.X_FSM.Current_state_x <= 4) and person.CA_state==1 and ca_conf==1 and person.human_pos and(((person.total_poss_conf!=0 and(person.pass_confidence/person.total_poss_conf) >= 0.5) or person.pass_confidence ==1) ):
            person.passingtext="Passing"
            person.passing=True

        elif person.statex==1 and person.CA_state==1 and person.human_pos and(((person.total_poss_conf!=0 and(person.pass_confidence/person.total_poss_conf) >= 0.5) or person.pass_confidence==1)):
            person.passingtext="Passing"
            person.passing=True


        elif (person.total_poss_conf!=0 and(person.pass_confidence/person.total_poss_conf) >= 0.9 and person.CA_state==1 and person.not_appearing ==False ):
            person.passingtext="Passing"
            person.passing=True

        else:
            person.passingtext="NotPassing"
            person.passing=False
        if person.yaw >34:
            person.headtext ="looking"
        else:
            person.headtext ="Notlooking"

            
        person.lost=False
        person.occulsion=False
        person.not_appearing=False
        person.yaw = 0 
        person.pitch = 0
        person.pass_confidence=0
        person.total_poss_conf=0
        prediction=0
    
    visualize(frame_persons,visualize_image)
    remove_disappearing_person(frame_persons)
    lost_persons_edit()
    reset_added_flag()
    remove_person_10_frames(counter)
    counter+=1
    end_time = time.time()
    print("-------------------------------------------","total time taken :",end_time-start_time,"---------------------------------------------------------------")
    output.write(str("-----------------------------------------------------------------------------------------------------------------"))
    output.write("\n")
total_end =time.time()
output.close()
print("total time taken :" ,total_end-total_start)
fault_precent = (faulty_predictions/total_predictions)*100
print("model faulty predictions precentage :",fault_precent,"%" )
            
            
    # Nose = 0
    # Neck = 1
    # RShoulder = 2
    # RElbow = 3
    # RWrist = 4
    # LShoulder = 5
    # LElbow = 6
    # LWrist = 7
    # RHip = 8
    # RKnee = 9
    # RAnkle = 10
    # LHip = 11
    # LKnee = 12
    # LAnkle = 13
    # REye = 14
    # LEye = 15
    # REar = 16
    # LEar = 17
    # Background = 18
                
           
    
   
    

    
        