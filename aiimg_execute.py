from flask import Flask, jsonify, request,send_file
import random
import shutil
import base64
import os
import json
import io
from datetime import datetime
import numpy as np
from upc.common import general, common
import subprocess
import requests
from operator import itemgetter
from celery import Celery
from redis import Redis
import time
import cv2

from PIL import Image
from os.path import join, basename,isfile,isdir
from datetime import timedelta

app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379/0',
    CELERY_RESULT_BACKEND='redis://localhost:6379/0',
    CELERY_TASK_ALWAYS_EAGER=False
)
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])

worker_process = None

redis = Redis()
# region Init
ROOT = r"/var/www/html/aiimg_execute"
DATA_ROOT = r"/var/www/html/aiimg_execute/web_dataset/Tokyo"
MODEL_ROOT = r"/var/www/html/aiimg_execute/model"

TOP_DATASET_P = join(DATA_ROOT, 'top')
SIDE_DATASET_P = join(DATA_ROOT, 'side')
SECOND_DATASET_P = join(DATA_ROOT, 'second_side')

TEST_TOP_P = join(DATA_ROOT, 'test', 'top')
TEST_SIDE_P = join(DATA_ROOT, 'test', 'side')
TEST_SECOND_SIDE_P = join(DATA_ROOT, 'test', 'second_side')

BG_P_TOP = join(ROOT, "upc", "data", 'background', 'top_bg')
BG_P_SIDE = join(ROOT, "upc", "data", 'background', 'side_bg')

MAIN_P = [TOP_DATASET_P, SIDE_DATASET_P, SECOND_DATASET_P]
TEST_P = [TEST_TOP_P, TEST_SIDE_P, TEST_SECOND_SIDE_P]
REGISTERING_STATE='registering'
REMOVING_STATE='removing'
MODEL_LIST=['top','side','second_top','second_side']
DEACTIVATED_STATE='deactivated'
ACTIVATE_STATE='activate'

# API_URL = 'https://ucloud.upc.co.jp/aiimg/api'
API_URL = 'http://192.168.11.43:5000/aiimg/api'

# endregion


@celery.task
def process_request(jobs):
    # Set folder save model by job name
    model_name= None
    try:
        save_to = jobs['jobs_name']
        url = join(API_URL, 'update_jobs_status')

        now = datetime.now().strftime('%Y%m%d%H%M%S')
        print("GET DATA FROM lIST PRODUCT TO AUTO CREATE ANNOTATION ")
        data_ = get_data_from_api(URL=join(API_URL, 'get_product_manager'))
        print("GET DATA OK --->  CREATE IMAGES WITH AUTO ANNOTATION")
        create_data_remove(data_, date_=now)
        
        # Update jobs status to running
        print("::POST UPDATE JOB DATA --->")
        data_json = {'status': 'running', 'jobs': jobs}
        response = requests.post(url, json=data_json)
        print("::POST UPDATE JOB DATA >>>>>> OK ")
        with open('/var/www/html/aiimg_execute/model/model_manager.json', 'r+', encoding='utf-8') as f:
                        model_manager = json.load(f)
        print("READ MODEL MANAGER --->")
        # loop task in job and run task by task
        for task_ in jobs['task_in_jobs']:
            if 0 == 0:
                print(f"{task_['model'].upper()} TRAINING")
                

                task_['start_train_time'] = datetime.now().strftime('%Y%m%d%H%M%S')

                # call api update job status from AIIMG

                data_json = {'status': 'update', 'jobs': jobs}
                response = requests.post(url, json=data_json)
                name = ""
                #
                if task_['type'] == 'new':
                    cmd, pretrain_path,name = train(model=task_['model'], pretrain=False, save_to=save_to, epochs=1,time_=0)
                    model_name=name
                    process = subprocess.check_call(cmd, shell=True)
                    cmd, pretrain_path, name= train(model=task_['model'],pretrain=True,pretrain_path=pretrain_path, save_to=save_to, epochs=1, time_=1)
                    model_name=name
                    process = subprocess.check_call(cmd, shell=True)

                else:
                    pretrain_path="''"
                    if len(model_manager[f"{task_['model']}"]) >0:
                        for weight in model_manager[f"{task_['model']}"]:
                            if weight['status']=='1_current':
                                pretrain_path=join(MODEL_ROOT,'base',weight['name'],'weights','best.pt')
                                break
                    cmd, pretrain_path, name= train(model=task_['model'], pretrain=True,pretrain_path=pretrain_path, save_to=save_to, epochs=1)
                    model_name=name
                    process = subprocess.check_call(cmd, shell=True)
                
                # remove first model trained if task_['type'] == new
                if os.path.isdir(join(MODEL_ROOT, 'base', f"{name[:-4]}_pre")):
                    shutil.rmtree(join(MODEL_ROOT, 'base', f"{name[:-4]}_pre"))

                # if dir empty=> remove

                task_['end_train_time'] = datetime.now().strftime('%Y%m%d%H%M%S')
                task_['status'] = 'OK'
               
                
                path=fr"{name}\weights\best.pt"
                uncheck_model = {
                    "type": task_['model'],
                    "name": f"{name}",
                    "path": path,
                    "date_registered": task_['start_train_time'],
                    "date_trained": task_['end_train_time'],
                    "mode": task_['mode'],
                    "images_num": None,
                    "model_size": 0,
                    "status": "3_uncheck",
                    "result": "--",
                    "desc": {}
                }
                model_manager[f"{task_['model']}"].append(uncheck_model)

                with open('/var/www/html/aiimg_execute/model/model_manager.json', 'w+', encoding='utf-8') as f:
                    json.dump(model_manager, f)

                # call api update job status from AIIMG
                data_json = {'status': 'update', 'jobs': jobs}
                response = requests.post(url, json=data_json)
                print(f"{task_['model'].upper()} TRAINED")

        redis.set(jobs['jobs_id'], str(jobs))
        # call api update job status from AIIMG
        data_json = {'status': 'finished', 'jobs': jobs}
        response = requests.post(url, json=data_json)

    except Exception as e:

        print(f"TRAIN ERROR : {e}")
        if model_name and  os.path.isdir(model_name):
            shutil.rmtree(
                    join(MODEL_ROOT, 'base',
                                f'{model_name}'))
        data_json = {'status': 'failed', 'jobs': jobs}
        response = requests.post(url, json=data_json)
    return 'TASK FINISH'

def train(model='', save_to="", pretrain=None,pretrain_path="''", epochs=150, batch_size=32, time_=None):

   

    train_path = "/home/upc/WorkSpaces/nam/yolov5/train.py"
    batch_size = 16 if model in['top','side'] else 8

    data_yaml_path = join(DATA_ROOT, f"{model}_dataset/custom_dataset.yaml")

    config_yaml_path = join(DATA_ROOT, f"{model}_dataset/custom_model.yaml")
    model_id = '01' if model == 'top' else '02' if model == 'side' else '03'

    save_path = join(MODEL_ROOT, "base")


    if pretrain:   
        name = f"{model_id}_{save_to}_cur"
    if time_ == 0:   
        name = f"{model_id}_{save_to}_pre"
    if time_ == 1:
        name = f"{model_id}_{save_to}_cur"
           
    command = f"python -m torch.distributed.run --nproc_per_node 2 {train_path} --batch {batch_size} --data {data_yaml_path} --cfg {config_yaml_path} --weights {pretrain_path} --epoch {epochs} --device 0,1 --project {save_path} --name {name}"
    weights_path=join(save_path,f'{name}','weights','best.pt')
    return command, weights_path, name

def remove():
    pass

@celery.task
def delete_completed_tasks():
    celery.backend.cleanup()


celery.conf.beat_schedule = {
    'delete-completed-tasks-every-30-minutes': {
        'task': 'delete_completed_tasks',
        'schedule': timedelta(minutes=30),
    },
}


@app.route('/ubuntu_api/tasks', methods=['GET'])
def get_active_tasks():
    i = celery.control.inspect()
    active_tasks = i.active()
    scheduled_tasks = i.scheduled()
    reserved_tasks = i.reserved()

    tasks = {
        'active': active_tasks,
        'scheduled': scheduled_tasks,
        'reserved': reserved_tasks
    }

    return jsonify(tasks)


@app.route('/ubuntu_api/clear', methods=['GET'])
def clear():
    keys = redis.keys('*')
    for key in keys:
        redis.delete(key)
    return "OK"


@app.route('/ubuntu_api/add_train_jobs', methods=['POST'])
def add_train_task():
    if request.method == 'POST':
        jobs = request.get_json()
        process_request.apply_async(kwargs={'jobs': jobs}, task_id=jobs['jobs_id'])
        return "Add ok"


@app.route('/ubuntu_api/cancel_task', methods=['POST'])
def cancel_train_jobs():
    if request.method == 'POST':
        json_ = request.get_json()
        celery.control.revoke(json_['jobs_id'], terminate=True)
        return "Remove ok"


@app.route('/ubuntu_api/result', methods=['POST'])
def get_result():
    tasks = redis.keys("*")
    value = []
    for rs in tasks:
        rss = redis.get(rs)
        value.append(rss)

    return jsonify({'rs': value})


@app.route('/ubuntu_api/converse', methods=["GET"])
def converse():
   side_, _ = general.get_all_file(join(DATA_ROOT, 'second_side_jpg'))
   converse2jpg(side_)
   return "OK"


@app.route('/ubuntu_api/auto_anno', methods=["GET"])
def auto_anno():
    data_ = get_data_from_api(URL=join(API_URL, 'get_product_manager'))
    print("GET DATA OK --->  CREATE IMAGES WITH AUTO ANNOTATION")
    create_data_remove(data_, date_=datetime.now().strftime('%Y%m%d%H%M%S'))
   
    return "OK"

@app.route('/ubuntu_api/auto_create_dataset', methods=["GET"])
def auto_create_dataset():
    for model in ['top','side','second_side']:
        create_dataset(join(DATA_ROOT,f'{model}_dataset','auto'),join(DATA_ROOT,f'{model}_dataset','uncheck'),without_valid=True)
    return "OK"


# region process

def create_dataset(source_dir, create_to_dir, without_valid=False):
    path_img_train, path_img_valid = join(create_to_dir, 'images', 'train'), join(create_to_dir, 'images', 'valid')
    path_lbl_train, path_lbl_valid = join(create_to_dir, 'labels', 'train'), join(create_to_dir, 'labels', 'valid')
   
    file_im_list = []
    for f in os.listdir(source_dir):
        if f.endswith('png') or f.endswith('jpg'):
            file_im_list.append(f)
    if without_valid:
        for f in file_im_list:
            shutil.move(join(source_dir, f), path_img_train)
            shutil.move(join(source_dir, f[:-4] + '.txt'), path_lbl_train)

            #second side has not for design
            # if isfile(join(source_dir, f[:-4] + '-for_ds.txt')):
            #     shutil.move(join(source_dir, f[:-4] + '-for_ds.txt'), path_lbl_for_design)

    else:
        ##TODO FIX THIS
        num_file = len(file_im_list)
        num_file_train = int(num_file * 0.8)
        num_file_valid = num_file - num_file_train

        random.shuffle(file_im_list)
        file_img_list_train = file_im_list[:num_file_train]
        file_img_list_valid = file_im_list[num_file_valid:]

        for f in file_img_list_train:
            shutil.move(join(source_dir, f), path_img_train)
            shutil.move(join(source_dir, f[:-4] + '.txt'), path_lbl_train)

        for f in file_img_list_valid:
            shutil.move(join(source_dir, f), path_img_valid)
            shutil.move(join(source_dir, f[:-4] + '.txt'), path_lbl_valid)


def base642PIL(base64_str):
    """
    It takes a base64 string, decodes it, and returns a PIL image object

    :param base64_str: The base64 string of the image
    :return: A PIL image object.
    """
    imgdata = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(imgdata)).convert("RGBA")
    return img

def u_path(path):
    return path.replace("\\", "/")

def remove_image(removing_products,model=None):
    print('DO REMOVE')
    with open(join(DATA_ROOT,f'{model}_dataset',f'{model}_dataset.json'),'r',encoding='utf-8') as f:
        dataset=json.load(f)

    backup_json=[]
    bk_path= join(DATA_ROOT,f"{model}_dataset","backup")
    os.makedirs(bk_path,exist_ok=True)
    print('DO REMOVE1')
    try:
        with open(join(bk_path,'backup.json'),'r+',encoding='utf-8') as f:
            backup_json=json.load(f)
    except IOError:
        pass
    rm_products=[]
    print('DO REMOVE2')
    for product in removing_products:
        for design in product['design']:
            if design['status']['uncheck']==REMOVING_STATE:
                rm_id=int(product['class_id'])
                rm_ds=int(design['design_id'])
                if len(rm_products)>0:
                    for rm_ in rm_products:
                        if rm_id == int(rm_['rm_id']) and rm_ds not in rm_['rm_ds'] :
                            rm_['rm_ds'].append(rm_ds)
                        else:
                            rm_products.append({
                                'rm_id':int(rm_id),
                                'rm_ds':[rm_ds]
                            })
                        break
                else:
                    rm_products.append({
                        'rm_id':int(rm_id),
                        'rm_ds':[rm_ds]
                    })
                    


    #backup
    
    print('DO BK')
   
    bk_images=[] 

   # mode is train or  valid 
    for image in dataset:
        for annotation in image['annotation']:
            if int(annotation['class_id']) in [rm_['rm_id'] for rm_ in rm_products]:
                print(annotation['class_id'])
                for rm in rm_products:
                    if int(rm['rm_id']) == int(annotation['class_id']):
                        for box in annotation['boxes']:
                            if int(box['design']) in rm['rm_ds']:
                                    image['modifi_date']=datetime.now().strftime('%Y%m%d%H%M%S')
                                    box['status']=DEACTIVATED_STATE
                    # Change status of [id] to deactivated if all [design] has been deactivated
                        status_state=[box['status'] for box in annotation['boxes']]
                        if ACTIVATE_STATE not in status_state:
                            annotation['status']=DEACTIVATED_STATE
                            image['modifi_date']=datetime.now().strftime('%Y%m%d%H%M%S')
                        bk_images.append(image)
                        ###########

                        #backup
                        # check exist backup
                        
                        new_backup_path=join(bk_path,image['image_name'])
                        new_back_path_to_save=fr"{model}_dataset\backup\{basename(new_backup_path)}"
                        if u_path(image['image_path']) is not None and u_path(image['image_path'])==new_back_path_to_save and isfile(new_backup_path):
                            break
                        else:
                            shutil.copy(join(DATA_ROOT,u_path(image['image_path'])),new_backup_path)
                            image['backup_path']=new_back_path_to_save
                            break
                
    print('DO BK2')
    for rm_image in bk_images:
        if ACTIVATE_STATE not in [ anno['status'] for anno in rm_image['annotation']]:
            if isfile(join(DATA_ROOT,u_path(rm_image['image_path']))):
                os.remove(join(DATA_ROOT,u_path(rm_image['image_path'])))
            if isfile(join(DATA_ROOT,u_path(rm_image['label_path']))):
                os.remove(join(DATA_ROOT,u_path(rm_image['label_path'])))
        else:
            #Read Opencv image
            image=cv2.imread(join(DATA_ROOT,u_path(rm_image['image_path'])),cv2.IMREAD_UNCHANGED)
            with open(join(DATA_ROOT,u_path(rm_image['label_path'])),'w+',encoding='utf-8') as f:
                for annotation in rm_image['annotation']:
                    # Check if id not is a rm_id and status is active
                    # rewrite it to label
                    if int(annotation['class_id']) not in [rm['rm_id'] for rm in rm_products] and annotation['status'] == ACTIVATE_STATE:

                        for box in annotation['boxes']:
                            if box['status']==ACTIVATE_STATE:
                                line_= f"{annotation['class_id']} {box['x_center']} {box['y_center']} {box['width']} {box['height']}"
                                f.write(line_)
                                f.write('\n')
                    if int(annotation['class_id']) in [rm['rm_id'] for rm in rm_products]:
                        # if id in remove id , and status is deactivate
                        if annotation['status']==DEACTIVATED_STATE:
                            #replace image in box by white box
                            for box in annotation['boxes']:
                                print('DO BK2 2')
                                box_=[box['x_center'],box['y_center'],box['width'],box['height']]
                                x,y,w,h=general.yolo_box_to_rec_box(box_,image.shape[:2])
                                cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,255),-1)

                            
                        if annotation['status']==ACTIVATE_STATE:
                            for box in annotation['boxes']:
                                print('DO BK2 3')
                                if box['status']==ACTIVATE_STATE:
                                    line_= f"{annotation['class_id']} {box['x_center']} {box['y_center']} {box['width']} {box['height']}"
                                    f.write(line_)
                                    f.write('\n')
                                if box['status']==DEACTIVATED_STATE:
                                    box_=[box['x_center'],box['y_center'],box['width'],box['height']]
                                    x,y,w,h=general.yolo_box_to_rec_box(box_,image.shape[:2])
                                    cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,255),-1)
            

            cv2.imwrite(join(DATA_ROOT,u_path(rm_image['image_path'])),image)
        
    print('DO BK3')
    list_im=[bk['image_name'] for bk in bk_images]
    if len(backup_json)>0:
        for backup in backup_json[:]:
            if backup['image_name'] in list_im:
                backup_json.remove(backup)
    backup_json.extend(bk_images)
        
    with open(join(bk_path,'backup.json'),'w+',encoding='utf-8') as f:
          json.dump(backup_json,f)

    for image in dataset[:]:
        status_state=[ano['status'] for ano in image['annotation']]
        if ACTIVATE_STATE not in status_state:
            dataset.remove(image)

    with open(join(DATA_ROOT,f'{model}_dataset',f'{model}_dataset.json'),'w',encoding='utf-8') as f:
        json.dump(dataset,f)
    

    print('REMOVE BOX IN IMAGES') 
    

def create_data_remove(data, date_=None):
    # TODO  run with product type
    # Add mode for single case pack, feature
    print('START CREATE DATA REMOVE') 
    registing_products={
        "top":[],
        "side":[],
        "second_top":[],
        "second_side":[] 
        }
    removing_products={
        "top":[],
        "side":[],
        "second_top":[],
        "second_side":[] 
        }
    
    for model in MODEL_LIST:
        for product in data[f'{model}']:
            for design in product['design']:
                if design['images'] is None or len(design['images']) == 0:
                    continue
                if design['status']['uncheck'] == REGISTERING_STATE:
                   registing_products[f'{model}'].append(product)

                if design['status']['uncheck'] == REMOVING_STATE:
                    removing_products[f'{model}'].append(product)
                     
    print('REMOVE OK') 
    for model in MODEL_LIST:
        if len(removing_products[f'{model}'])>0:
            remove_image(removing_products[f'{model}'],model=f'{model}') 

        if len(registing_products[f'{model}'])>0:
               resize_and_merge(registing_products[f'{model}'], join(DATA_ROOT,f'{model}_dataset','auto'), model=f'{model}', im_num=10, date_=date_)
               print('CREATE AUTO IMAGE OK') 
               create_dataset(join(DATA_ROOT,f'{model}_dataset','auto'),join(DATA_ROOT,f'{model}_dataset','base'),without_valid=True)
               print('CREATE AUTO DATASET OK') 
               create_dataset_manager(model=model,mode='auto')
               print('CREATE AUTO DATASET OK 2')        
       
def create_dataset_manager(model,mode=None):
        with open(join(DATA_ROOT,f'{model}_dataset',f'{model}_dataset.json'),'r',encoding='utf-8') as f:
            dataset_manager=json.load(f)
        path_= join(DATA_ROOT,f'{model}_dataset','base')
        list_images=[image['image_name'] for image in dataset_manager]
        all_files,_= general.get_all_file(path_)
        for f in all_files:
            if basename(f) not in list_images:
               
                image={
                    "image_name":basename(f)
                }
                if f.endswith('png') or f.endswith('jpg'):
                    type_='train' if 'train' in f else 'valid'
                    image_path_to_save=fr"{model}_dataset\base\images\{type_}\{basename(f)}"
                    image['image_path']=image_path_to_save

                    label_path_to_save=image_path_to_save.replace('images','labels')
                    label_path_to_save=label_path_to_save[:-4]+'.txt'
                    image['label_path']=label_path_to_save
                    image['backup_path']=None
                    image['dependent_by']='uncheck'
                    image['create_by']=mode
                    image['regist_date']=datetime.now().strftime("%Y%m%d%H%M%S")
                    image['modifi_date']=datetime.now().strftime("%Y%m%d%H%M%S")
                    image['annotation']=[]

                    label_path=f.replace('images','labels')
                    label_path=label_path[:-4]+'.txt'
                    if isfile(label_path):
                        with open(label_path, 'r') as f:
                            lines=[line.rstrip() for line in f.readlines()]
                        for line in lines:
                            liness_=line.split(' ')
                            class_id_list=[anno['class_id'] for anno in image['annotation']]
                            if int(liness_[0]) in class_id_list:
                                for anoo in image['annotation']:
                                    if int(anoo['class_id']) == int(liness_[0]):
                                        position={
                                            "x_center":float(liness_[1]),
                                                "y_center":float(liness_[2]),
                                                "width":float(liness_[3]),
                                                "height":float(liness_[4]),
                                                "design":0,
                                                "status":"activate"
                                        }
                                        anoo['boxes'].append(position)
                                
                                        break
                            else:
                                anoo_box={
                                        "class_id": int(liness_[0]),
                                        "boxes":[{
                                                "x_center":float(liness_[1]),
                                                "y_center":float(liness_[2]),
                                                "width":float(liness_[3]),
                                                "height":float(liness_[4]),
                                                "design":0,
                                                "status":"activate"
                                        }],
                                        "status":"activate"
                                        }
                                image['annotation'].append(anoo_box)
        
                        dataset_manager.append(image)   
        with open(join(DATA_ROOT,f'{model}_dataset',f'{model}_dataset.json'),'w',encoding='utf-8') as f:
            json.dump(dataset_manager,f)


def resize_and_merge(registing_products, save_to, bg_p=None, model=None, im_num=5, date_=None):
    # For top side model
    if 'second' not in model: 
    # Get list of images be resized to
        [min_size,max_size] = [0.9,1.1] if model=='top' else [0.7,1.2]
        resized_images = common.resize(registing_products, min_size=min_size, max_size=max_size, step=0.1, im_num=im_num,DATA_ROOT=DATA_ROOT)

        bg_images_path, _ = general.get_all_file(join(ROOT, "upc", "data", 'background', f'{model}_bg'))

        # Get list of background images with PIL Image format

        bg_images = [Image.open(im_p).convert("RGBA") for im_p in bg_images_path]

        # Get merged list of images be resized
        common.merge_thread(foregrounds=resized_images,
                            backgrounds=bg_images,
                            p_save=save_to,
                            rotate_=359 if model == 'top' else 5,
                            rotate_step=10 if model == 'top' else 1,
                            cutout_=True,
                            name=f"{date_}"
                            )
    else : # for second model 
        for product in registing_products:
            for design in product['design']:
                for idx,image_ in enumerate(design['images']):
                    name = f"{product['class_id']}_{design['design_id']}-{str(idx)}"
                    image_path =image_['image_path'].replace('\\','/')
                    im= Image.open(join(DATA_ROOT,image_path)).convert('RGB')
                    im.save(join(save_to,name+'.jpg'))
                    x, y, x1, y1 = image_['feature_position']['x_min'],image_['feature_position']['y_min'],image_['feature_position']['x_max'],image_['feature_position']['y_max']
                    h,w=image_['size']['height'],image_['size']['width']
                    xywh = [x, y, x1 - x, y1 - y]
                    pos = bnd_box_to_yolo_box(xywh, [h, w])
                    with open(join(save_to,name) + ".txt", 'a') as f:
                    
                        f.write(f"{product['class_id']} {pos[0]} {pos[1]} {pos[2]} {pos[3]}\n")
        

def bnd_box_to_yolo_box(box, img_size):
    (x_min, y_min) = (box[0], box[1])
    (w, h) = (box[2], box[3])
    x_max = x_min + w
    y_max = y_min + h

    x_center = float((x_min + x_max)) / 2 / img_size[1]
    y_center = float((y_min + y_max)) / 2 / img_size[0]

    w = float((x_max - x_min)) / img_size[1]
    h = float((y_max - y_min)) / img_size[0]

    return x_center, y_center, w if w < 1 else 1.0, h if h < 1 else 1.0


def get_images_for_second_side(images, data,date_=""):
    if data['status']=="registering":
        for idx_, im_base64 in enumerate(data['im_base64']):
            im = base642PIL(im_base64['im_base64'])
            w, h = im.size
            x, y, x1, y1 = itemgetter(0, 1, 2, 3)(im_base64['feature_pos'])
            xywh = [x, y, x1 - x, y1 - y]
            pos = bnd_box_to_yolo_box(xywh, [h, w])
            image = {'im': im, 'feature_pos': pos, 'class_id': data['class_id'], 'insert_date': date_,'idx':idx_}
            images.append(image)


def get_images(images, product_ds, class_id=None):
    # TODO check datetime
    for idx_, im_base64 in enumerate(product_ds['image_base64']):
        im = base642PIL(im_base64['im_base64'])
        name = f"{class_id}_{product_ds['design_id']}-{str(idx_)}"
        image = {name: im}
        images.append(image)


def remove_design():
    pass


def update_status(model=""):
    try:
        data = {'model': model, 'status': 'OK'}
        response = requests.post(url=join(API_URL, 'update_status'), json=data,
                                 verify=False)
        if response.text == 'OK':
            print('UPDATE STATUS OK')
        else:
            print('UPDATE STATUS FAIL')
    except Exception as e:
        print('UPDATE STATUS ERROR ')


def run_with_real_image(data):
    for im_info in data:
        if not im_info['STATUS']:
            # TODO check datetime
            im = base642PIL(im_info['IMAGE_BASE64'])
            name = im_info['NAME']
            if im_info['MODEL'] == 'TOP':
                save_path = f"/home/upc/WorkSpaces/nam/yakult_project/aiimg_execute/upc/data/web_dataset/top"
            elif im_info['MODEL'] == 'SIDE':
                save_path = f"/home/upc/WorkSpaces/nam/yakult_project/aiimg_execute/upc/data/web_dataset/side"
            else:
                save_path = f"/home/upc/WorkSpaces/nam/yakult_project/aiimg_execute/upc/data/web_dataset/case"
            im.save(join(save_path, 'images', 'train', name))
            with open(join(save_path, 'labels', 'train', name[:-4] + '.txt'), 'a') as f:
                for k, v in im_info['ANNOTATION'].items():
                    for pos in v:
                        f.write(f"{k} {pos[0]} {pos[1]} {pos[2]} {pos[3]}\n")
            #TODO set design
            # with open(join(save_path, 'labels', 'for_design', name[:-4] + '-for_ds.txt'), 'a') as f:
            #     for k, v in im_info['ANNOTATION'].items():
            #         for pos in v:
            #             f.write(f"{k} {pos[0]} {pos[1]} {pos[2]} {pos[3]} {pos[4]}\n")


def get_data_from_api(URL=None):
    r = requests.get(url=URL)
    try:
        data = r.json()
        return data
    except Exception as e:
        print(e)
        return None


def upload_model(model='', url="https://ucloud.upc.co.jp/cloud/api/ApiFile/PostFiles"):
    list_model = os.listdir("/home/upc/WorkSpaces/nam/yakult_project/checked_model")
    model_id = '01' if model == 'top' else '02' if model == 'side' else '03'
    if len(list_model) > 0:
        lasted = np.max([int(date) for date in list_model])
        weight = f"/home/upc/WorkSpaces/nam/yakult_project/checked_model/{lasted}/{model_id}_{lasted}/weights/best.pt"
        if isfile(weight):
            with open(weight, 'rb') as f:
                files = {"data": (None, json.dumps({
                    "FILE_TYPE": "2",
                    "NAME": f"model_{model}",
                    "FILE_NAME": f"NSJ_0{model_id}.pt",
                    "FILE_PATH": "C:\\UPC\\uscan\\data\\assets\\models",
                    # "RELEASE_DATE": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                    "RELEASE_DATE": None,
                    "MEMO": "TEST"}), "application/json"),
                         "FILE": (f"NSJ_0{model_id}.pt", f, "application/octet-stream")
                         }
                response = requests.post(url=url, files=files,
                                         verify=False)
            print(response)
            update_status(model=model)
        else:
            print("MODEL FILE IS NOT EXIST TO UPLOAD")


def converse2jpg(path):
    for f in path:
        if f.endswith('png'):
            im1 = Image.open(f).convert("RGB")
            im1.save(f[:-4] + '.jpg')
            os.remove(f)


@app.route('/')
def aiimg_execute():
    return 'This is Flask Server for train Yolov5 '


# endregion

if __name__ == '__main__':
    app.run()
