from flask import Flask, jsonify, request,send_file
import random
import shutil
import base64
import os
import json
import uuid
import io
from datetime import datetime
import numpy as np
from upc.common import general

from urllib.parse import quote_plus, urlparse


import albumentations as A
import subprocess
import threading
import requests
from operator import itemgetter
from celery import Celery
from redis import Redis
import time
import cv2

from PIL import Image
from os.path import join, basename,isfile,isdir
from datetime import timedelta

from pymongo import MongoClient
from pymongo.errors import OperationFailure
from bson import DBRef,ObjectId

app = Flask(__name__)

username = "aiimg"
password = "Kr3H4q=h_xn(D{ObdzFv"
escaped_username = quote_plus(username)
escaped_password = quote_plus(password)


# Connect to the MongoDB server running at the specified IP and port
client = MongoClient(f"mongodb://{escaped_username}:{escaped_password}@aiimg01.upc.co.jp:27017/?authSource=admin")
# client = MongoClient("mongodb://192.168.0.111:27017/")

# Access the "aiimg" database
db = client['aiimg_104']

# Access different collections within the "aiimg" database
users_coll = db["users"]               # Collection for user data
products_coll = db["products"]         # Collection for product data
jobs_coll = db["jobs"]                 # Collection for job data
dataset_coll = db["datasets"]           # Collection for dataset data
actual_im_coll = db["actual_images"]   # Collection for actual images
keys_coll = db["keys"]                 # Collection for keys
models_coll = db["models"]               # Collection for model



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
ROOT1 = r"/var/www/html/aiimg_execute/Data"

BG_P_TOP = join(ROOT, "upc", "data", 'background', 'top_bg')
BG_P_SIDE = join(ROOT, "upc", "data", 'background', 'side_bg')

REGISTER_INTRAIN_STATE='register-intrain'
REMOVING_STATE='remove-intrain'
MODEL_LIST=['top','side','second_top','second_side']
DEACTIVATED_STATE='deactivated'
ACTIVATE_STATE='activate'

# endregion

def get_dbref(user_,key):
    user = users_coll.find_one({"username": user_})
    if user:
        db_ref = user[key]
        return db_ref
    else:
        return None


def get_user_id(username):
    # Retrieve the user document based on the current user's username if they are authenticated; otherwise, set to None.
    # 現在のユーザーが認証されている場合、ユーザー文書を現在のユーザーのユーザー名を基に取得します。認証されていない場合、Noneに設定されます。
    user_ = users_coll.find_one({"username": username})
    return user_["_id"] if user_ else None

def get_user_name(user_id):
    user_ = users_coll.find_one({"_id":user_id})
    return user_["username"] if user_ else None


def remove_model_from_physical_drive(username,model_filter,category):
    model_doc= models_coll.find_one(model_filter)
  
    if model_doc:
        model_dir=join(ROOT1,username,category,'Model', 'base', f"{model_doc['name']}")
        shutil.rmtree(model_dir) if isdir(model_dir) else None

def remove_dataset_from_physical_drive(username,dataset_filter,category):
    cursor=dataset_coll.find(dataset_filter)
    remove_images=list(cursor)
    for im in remove_images:
        im_path=join(ROOT1,username,category,'Dataset', u_path(im['image_path']))
        lb_path=join(ROOT1,username,category,'Dataset', u_path(im['label_path']))
        os.remove(im_path) if isfile(im_path) else None
        os.remove(lb_path) if isfile(lb_path) else None


def reverse_removed_images(username,model,category):

       # Find classes with 'status.uncheck' set to 'REMOVING_STATE'
        # 'status.uncheck' が 'REMOVING_STATE' に設定されたクラスを検索します
        cursor = products_coll.find({'user_id': get_user_id(username),
                                     'category': category,
                                    'model_direction': model,
                                    'status.uncheck': REMOVING_STATE},
                                    {'_id': 0, 'class_id': 1})
        removing_cls_ids = list(cursor)
        removing_cls_ids = [int(cls['class_id']) for cls in removing_cls_ids]

      

        # Update the status of selected annotations and update mod_date
        # 選択した注釈のステータスを更新し、mod_dateを更新します
        dataset_coll.update_many({'user_id': get_user_id(username),
                                'annotation.class_id': {'$in': removing_cls_ids},
                                'category': category,
                                'model_direction': model},
                                {'$set': {'annotation.$[ano].status': ACTIVATE_STATE,
                                         
                                          'mod_date': datetime.now().strftime('%Y%m%d%H%M%S'),
                                           'status.uncheck':"registered"}},
                                array_filters=[{'ano.class_id': {'$in': removing_cls_ids}}])

        # Find reverse_images based on criteria
        # 基準に基づいてreverse_imagesを検索します

        cursor = dataset_coll.find({'user_id': get_user_id(username),
                                    'category': category,
                                    'model_direction': model,
                                    'annotation.class_id': {'$in': removing_cls_ids}})
        reverse_images = list(cursor)

        # Set the number of threads for image processing
        # 画像処理のためのスレッド数を設定します
        num_threads = 30  # You can set the number of threads here
        # Calculate the number of threads based on the number of backup images
        # バックアップイメージの数に基づいてスレッド数を計算します
        num_threads = min(len(reverse_images), num_threads)

        if num_threads > 0:
            # Use ThreadPoolExecutor to process images in parallel
            # 画像を並行して処理するためにThreadPoolExecutorを使用します
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Submit image processing tasks to the thread pool
                # 画像処理タスクをスレッドプールに送信します
                for im in reverse_images:
                    executor.submit(reverse_from_backup, im, username,category)
        
        dataset_coll.update_many({'user_id': get_user_id(username),
                                   'category': category,
                                   'model_direction': model,
                                   "annotation.status": {"$all": [ACTIVATE_STATE]},
                                    'annotation.class_id': {'$in': removing_cls_ids}},
                                    {'$set': {
                                            'backup_path': None,
                                            'mod_date': datetime.now().strftime('%Y%m%d%H%M%S'),
                                             'status.uncheck':"registered",
                                    }})

def remove_created_models(username,model,category):
        # Define a model filter for deletion
        # 削除のためのモデルフィルタを定義します
        model_filter = {'model_direction': model,
                        'category': category,
                        'status': 'uncheck',
                        'user_id': get_user_id(username)}

        # Remove the trained model from the physical drive if the job failed
        # ジョブが失敗した場合、物理ドライブからトレーニング済みモデルを削除します
        remove_model_from_physical_drive(username, model_filter,category)

        # Remove the model from the MongoDB collection
        # MongoDBコレクションからモデルを削除します
        models_coll.delete_one(model_filter)

def remove_created_images(username,model,category):
       # Define a filter for datasets created by 'auto'
        # 'auto'によって作成されたデータセットのフィルタを定義します
        create_by_auto_filter = {'user_id': get_user_id(username),
                                'category': category,
                                "status.old":None,
                                'status.current':None,
                                'status.uncheck':REGISTER_INTRAIN_STATE,
                                'create_by': 'auto',
                                'model_direction': model}

        # Remove the dataset from the physical drive
        # 物理ドライブからデータセットを削除します
        remove_dataset_from_physical_drive(username, create_by_auto_filter,category)

        # Delete the dataset from the MongoDB collection
        # MongoDBコレクションからデータセットを削除します
        dataset_coll.delete_many(create_by_auto_filter)

        # Define a filter for datasets created by 'actual'
        # 'actual'によって作成されたデータセットのフィルタを定義します
        create_by_actual_filter = {'user_id': get_user_id(username),
                                    'category': category,
                                    "status.old":None,
                                    'status.current':None,
                                    'status.uncheck':REGISTER_INTRAIN_STATE,
                                    'create_by': 'actual',
                                    'model_direction': model}

        # Remove the dataset from the physical drive
        # 物理ドライブからデータセットを削除します
        remove_dataset_from_physical_drive(username, create_by_actual_filter,category)

        # Find and get the image names created by 'actual'
        # 'actual'によって作成された画像名を検索および取得します
        cursor = dataset_coll.find(create_by_actual_filter)
        created_images_by_actual = [im['image_name'] for im in cursor]

        # Update the status of images created by 'actual' in the actual_im_coll
        # actual_im_coll内の 'actual'によって作成された画像のステータスを更新します
        actual_im_coll.update_many(
            {
                "user_id": get_user_id(username),
                "image_name": {'$in': created_images_by_actual},
            },
            {"$set": {"status": "added"}},
        )

        # Delete the dataset from the MongoDB collection
        # MongoDBコレクションからデータセットを削除します
        dataset_coll.delete_many(create_by_actual_filter)

def reverse_data(username,task,model, category):
    # Update the job status to 'failed' in the jobs collection
    # ジョブのステータスを 'failed' に更新する（jobsコレクション内）
    jobs_coll.update_one({'_id': task['_id']},
                        {
                            '$set': {'status': 'failed'},
                        })

    # Loop through the execute_model_list
    # execute_model_listをループします
    reverse_removed_images(username,model,category)
    remove_created_models(username,model,category)
    remove_created_images(username,model,category)


def init_uncheck_model(task,name,path):
      uncheck_model = {
                        "model_direction": task['model_direction'],
                        "name": f"{name}",
                        "path": path,
                        "reg_date": task['start_at'],
                        "trained_date": task['end_at'],
                        "release_date": None,
                        "mode": task['mode'],
                        "images_num": None,
                        "model_size": 0,
                        "status": "uncheck",
                        "result": "--",
                        "desc": {},
                        "category":task['category'],
                        "user_id":task['user_id'],

                        }
      return uncheck_model
@celery.task
def process_request(data):
    task_id=ObjectId(data['task_id'])
    task_filter={'_id':task_id}
    task_=jobs_coll.find_one(task_filter)
  
    try:
        print("Initiating")
        username=get_user_name(task_['user_id'])
        category=task_['category']
        save_to = task_['name']
        model=task_['model_direction']
        
        print("Create data remove")
        create_data_remove(username,category,save_to,model)
        print("Create data remove OK")

        jobs_coll.update_one({'_id':task_id},{'$set':{'status':'running'},})
        print(f"{model.upper()} TRAINING")
        jobs_coll.update_one(task_filter,{'$set':{'start_at':datetime.now().strftime('%Y%m%d%H%M%S')}})
        name = ""
        # raise Exception("test")
        #
        if task_['type'] == 'new':
            print("0")
            cmd, pretrain_path,name = train(username,category,model=model, pretrain=False, save_to=save_to, epochs=int(task_['epoch']),time_=0)
            
            process = subprocess.check_call(cmd, shell=True)

            cmd, pretrain_path, name= train(username,category,model=model,pretrain=True,pretrain_path=pretrain_path, save_to=save_to, epochs=int(task_['epoch']), time_=1)

            process = subprocess.check_call(cmd, shell=True)

        else:
            pretrain_path="''"
            if task_['for_model']=='uncheck': 
                print("1")
                models_doc = models_coll.find_one({'user_id':get_user_id(username),
                                                'model_direction':task_['model_direction'],
                                                'category':category,
                                                'status':'uncheck'},
                                                {'_id':0,'name':1}) 
                print(models_doc)
                pretrain_path=join(ROOT1,username,category,'Model','base',models_doc['name'],'weights','best.pt')
            elif task_['for_model']=='current':
                    models_doc = models_coll.find_one({'user_id':get_user_id(username),
                                                'model_direction':task_['model_direction'],
                                                'category':category,
                                                'status':'current'},
                                                {'_id':0,'name':1}) 
                    pretrain_path=join(ROOT1,username,category,'Model','base',models_doc['name'],'weights','best.pt')
        
        
            cmd, pretrain_path, name= train(username,category,model=model, pretrain=True,pretrain_path=pretrain_path, save_to=save_to, epochs=int(task_['epoch']))
            process = subprocess.check_call(cmd, shell=True)
    
        # remove first model trained if task_['type'] == new
        if os.path.isdir(join(ROOT1,username,category,'Model', 'base', f"{name[:-4]}_pre")):
            shutil.rmtree(join(ROOT1,username,category,'Model', 'base', f"{name[:-4]}_pre"))

        # if dir empty=> remove
        print("Calculating images count")
        cursor = products_coll.find(
            {"user_id":get_user_id(username),'category':category, "model_direction": model}, {"_id": 0}
        )
        products_doc = list(cursor)
        for product in products_doc:
            cursor = list(dataset_coll.find(
            {
                "user_id": get_user_id(username),
                "model_direction": model ,
                'category':category,
                "annotation.class_id": product['class_id'],
            }))
            im_count=len(cursor)

            products_coll.update_one({"user_id": get_user_id(username),
                "model_direction": model ,
                'category':category,
                "class_id": product['class_id']},
                {'$set':{'images_count':im_count}})
        
        print("Update task status")
        jobs_coll.update_one(task_filter,
                            {
                            '$set':{'end_at': datetime.now().strftime('%Y%m%d%H%M%S'),
                                    'status':'finished'}
                            })
        path=fr"{name}\weights\best.pt"
        
        #Updated
        task_=jobs_coll.find_one(task_filter)

        uncheck_model =init_uncheck_model(task_,name,path)
        
        models_coll.delete_one(
                        {"user_id":get_user_id(username),
                        'category':category,
                        'model_direction':model,
                        'status':'uncheck'
                            })
        models_coll.insert_one(uncheck_model)
        print(f"{model.upper()} TRAINED")
       
        return 'TASK FINISH'

    except Exception as e:

        print(f"TRAIN ERROR : {e}")
        reverse_data(username,task_,model, category)
        reverse_status(model,username,category)


def reverse_status(model,username,category):  
    products_coll.update_many(
            {
                "user_id": get_user_id(username),
                "model_direction": model,
                "category": category,
                "status.uncheck": REGISTER_INTRAIN_STATE,
            },
            {'$set':{"status.uncheck": "registering"}}
        )
    products_coll.update_many(
            {
                "user_id": get_user_id(username),
                "model_direction": model,
                "category": category,
                "status.uncheck": "remove-intrain",
            },
             {'$set':{"status.uncheck": "removing"}}
        )

def train(user,category,model='', save_to="", pretrain=None,pretrain_path="''", epochs=150, batch_size=32, time_=None,):


    train_path = "/home/upc/nam/yolov5/train.py"
    batch_size = 32 if model in['top','side'] else 16

    data_yaml_path = join(ROOT1,user,category,'Dataset', f"{model}_dataset/custom_dataset.yaml")

    config_yaml_path = join(ROOT1,user,category,'Dataset', f"{model}_dataset/custom_model.yaml")
    model_id = '01' if model == 'top' else '02' if model == 'side' else '03'

    save_path = join(ROOT1,user,category,'Model', "base")


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


@app.route('/api/add_train_task', methods=['POST'])
def add_train_jobs():
    if request.method == 'POST':
        data = request.get_json()
        process_request.apply_async(kwargs={'data': data})
        return "add ok"
   
   
   


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
    
def process_image(im,username,category):
        if ACTIVATE_STATE not in [anno['status'] for anno in im['annotation']]:
            if isfile(join(ROOT1,username, category, 'Dataset', u_path(im['image_path']))) and isfile(join(ROOT1,username,category, 'Dataset', u_path(im['label_path']))):
                os.remove(join(ROOT1,username,category, 'Dataset', u_path(im['image_path'])))
                os.remove(join(ROOT1,username,category, 'Dataset', u_path(im['label_path'])))
        else:
            # Read OpenCV image
            image = cv2.imread(join(ROOT1,username,category, 'Dataset', u_path(im['image_path'])), cv2.IMREAD_UNCHANGED)
            with open(join(ROOT1,username,category, 'Dataset', u_path(im['label_path'])), 'w+', encoding='utf-8') as f:
                for cls_id in im['annotation']:
                    if cls_id["status"] == ACTIVATE_STATE:
                        for box in cls_id['boxes']:
                                line_ = f"{cls_id['class_id']} {box['x_center']} {box['y_center']} {box['width']} {box['height']}"
                                f.write(line_)
                                f.write('\n')
                    if cls_id['status'] == DEACTIVATED_STATE:
                        for box in cls_id['boxes']:
                            box_ = [box['x_center'], box['y_center'], box['width'], box['height']]
                            x, y, w, h = general.yolo_box_to_rec_box(box_, image.shape[:2])
                            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)

            cv2.imwrite(join(ROOT1,username,category, 'Dataset', u_path(im['image_path'])), image)

def reverse_from_backup(im,username,category):
            if im['backup_path'] and  isfile(join(ROOT1,username,category, 'Dataset', u_path(im['backup_path']))):
                image = cv2.imread(join(ROOT1,username,category, 'Dataset', u_path(im['backup_path'])), cv2.IMREAD_UNCHANGED)
                with open(join(ROOT1,username,category, 'Dataset', u_path(im['label_path'])), 'w+', encoding='utf-8') as f:
                    for cls_id in im['annotation']:
                        if cls_id["status"] == ACTIVATE_STATE:
                            for box in cls_id['boxes']:
                                    line_ = f"{cls_id['class_id']} {box['x_center']} {box['y_center']} {box['width']} {box['height']}"
                                    f.write(line_)
                                    f.write('\n')
                        if cls_id['status'] == DEACTIVATED_STATE:
                            for box in cls_id['boxes']:
                                box_ = [box['x_center'], box['y_center'], box['width'], box['height']]
                                x, y, w, h = general.yolo_box_to_rec_box(box_, image.shape[:2])
                                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)

                cv2.imwrite(join(ROOT1,username,category, 'Dataset', u_path(im['image_path'])), image)
                if DEACTIVATED_STATE not in [anno['status'] for anno in im['annotation']]:
                    os.remove(join(ROOT1,username,category, 'Dataset', u_path(im['backup_path'])))




from concurrent.futures import ThreadPoolExecutor

def backup_remove(removing_cls_ids, model,username,category):
    bk_path= join(ROOT1,username,category,'Dataset',f"{model}_dataset","backup")
    os.makedirs(bk_path,exist_ok=True)

    dataset_coll.update_many({'user_id':get_user_id(username),
                            'annotation.class_id':{'$in':removing_cls_ids},
                            'category':category,
                            'model_direction':model},
                            {'$set':{'annotation.$[ano].status':DEACTIVATED_STATE}},
                            array_filters=[{'ano.class_id':{'$in':removing_cls_ids}}])
    
    dataset_coll.update_many({'user_id':get_user_id(username),
                            'annotation.class_id':{'$in':removing_cls_ids},
                            'category':category,
                            'model_direction':model,
                            "annotation.status": DEACTIVATED_STATE,
                            "annotation": {"$not": {"$elemMatch": {"status": ACTIVATE_STATE}}}},
                            {'$set':{'status.uncheck':"removed"}})
    
    cursor=dataset_coll.find({'user_id':get_user_id(username),
                                'category':category,
                               'model_direction':model,
                              'annotation.class_id':{'$in':removing_cls_ids}})
    bk_images=list(cursor)
    
    
    for image in bk_images:
        new_backup_path=join(bk_path,image['image_name'])
        new_back_path_to_save=fr"{model}_dataset\backup\{basename(new_backup_path)}"

        # check exist backup
        if u_path(image['image_path']) is not None and u_path(image['image_path'])==new_back_path_to_save and isfile(new_backup_path):
            continue

        else:
            shutil.copy(join(ROOT1,username,category,'Dataset',u_path(image['image_path'])),new_backup_path)
            dataset_coll.update_one({'user_id':get_user_id(username),
                                    'category':category,
                                    'model_direction':model,
                                    'image_name':image['image_name']},
                                    {'$set':{'backup_path':new_back_path_to_save}})

    # Process images concurrently using threading
    num_threads = 30  # You can set the number of threads here
    # Calculate the number of threads based on the number of backup images
    num_threads = min(len(bk_images), num_threads)

    if num_threads > 0:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit image processing tasks to the thread pool
            for im in bk_images:
                executor.submit(process_image, im,username,category)

    print(f'REMOVE BOX IN IMAGES FROM {model}')

def create_actual_image(username,model,category):
    cursor=actual_im_coll.find({'username': username,
                                'model_direction':model,
                                'category':category,
                                'status':'added'})
    actual_images=list(cursor)
    insert_images=[]


    for image in actual_images:
        im = Image.open(io.BytesIO( base64.b64decode(image['image_base64']))).convert("RGB")
    
        im_name=image['image_name']
        im_txt=image['image_name'][:-4]+'.txt'

        im.save(join(ROOT1,username,category,'Dataset',f'{model}_dataset','base','images','train',im_name))
        
        with open(join(ROOT1,username,category,'Dataset',f'{model}_dataset','base','labels','train',im_txt),'w+',encoding='utf-8') as f:
    
            for anno in image['annotation']:
                    if anno['status'] == ACTIVATE_STATE:
                        for box in anno['boxes']:
                                line_= f"{anno['class_id']} {box['x_center']} {box['y_center']} {box['width']} {box['height']}"
                                f.write(line_)
                                f.write('\n')

        image_={ "image_name":im_name,
                "image_path":fr"{model}_dataset\base\images\train\{im_name}",
                "label_path":fr"{model}_dataset\base\labels\train\{im_txt}",
                "backup_path": None,
                "status":{"old":None,
                          'current':None,
                          'uncheck':"registered"},
                "create_by": "actual",
                "reg_date":datetime.now().strftime("%Y%m%d%H%M%S"),
                "mod_date":datetime.now().strftime("%Y%m%d%H%M%S"),
                "annotation":image['annotation'],
                'model_direction':model,
                "user_id":get_user_id(username),
                "category":category }
        insert_images.append(image_)
    
    if len(actual_images)>0:
        actual_im_coll.update_many({'username': username,
                                    'model_direction':model,
                                    "category":category,
                                    'image_name': {'$in':[im['image_name'] for im in actual_images]}},
                                    {'$set':{'status':REGISTER_INTRAIN_STATE}})
    
    if len(insert_images) > 0:
        dataset_coll.insert_many(insert_images)

def create_data_remove(username,category,save_to,model):
 
    # TODO  run with product type
    # Add mode for single case pack, feature
    print('START CREATE DATA REMOVE') 
    create_actual_image(username,model,category)
    cursor=products_coll.find({'user_id': get_user_id(username),
                               "category": category,
                                'model_direction':model,
                                'status.uncheck':REGISTER_INTRAIN_STATE})
    registing_products=list(cursor)

    cursor=products_coll.find({'user_id': get_user_id(username),
                               'category':category,
                                'model_direction':model,
                                'status.uncheck':REMOVING_STATE},{'_id':0,'class_id':1})
    removing_cls_ids=list(cursor)
    removing_cls_ids=[int(cls['class_id']) for cls in removing_cls_ids]   
        
    backup_remove(removing_cls_ids,model,username,category) 
    resize_and_merge(username,category, registing_products,model, im_num=10)
    print(f'CREATE AUTO IMAGE FOR {model.upper()} OK') 

def resize_and_merge(username,category,registing_products, model=None, im_num=5):
    # For top side model
    if 'second' not in model: 
    # Get list of images be resized to
        [min_size,max_size] = [0.9,1.1] if model=='top' else [0.7,1.3]
        resized_images = resize(registing_products, min_size=min_size, max_size=max_size, step=0.1, im_num=im_num,DATA_ROOT=join(ROOT1,username,category,'Dataset'))

        bg_images_path, _ = general.get_all_file(join(ROOT, "upc", "data", 'background', f'{model}_bg'))

        # Get list of background images with PIL Image format

        bg_images = [Image.open(im_p).convert("RGBA") for im_p in bg_images_path]

        # Get merged list of images be resized
        merge_thread(foregrounds=resized_images,
                            backgrounds=bg_images,
                            model=model,
                            rotate_=359 if model == 'top' else 5,
                            rotate_step=10 if model == 'top' else 1,
                            cutout_=True,
                            DATA_TRANSFER={'DATA_ROOT':join(ROOT1,username,category,'Dataset'),
                                           'username':username,
                                           'category':category ,
                                           'user_id':get_user_id(username)}
                            )
    else : # for second model 
        for product in registing_products:
                for image_ in product['images']:
                    name = uuid.uuid4()

                    image_path =image_['image_path'].replace('\\','/')
                    im= Image.open(join(ROOT1,username,category,'Dataset',image_path)).convert('RGB')

                    txt_p= join(ROOT1,username,category,'Dataset',f'{model}_dataset','base','labels','train',f'{name}.txt')
                    txt_p_for_save= join(f'{model}_dataset','base','labels','train',f'{name}.txt')

                    im_p= join(ROOT1,username,category,'Dataset',f'{model}_dataset','base','images','train',f'{name}.jpg')
                    im_p_for_save=join(f'{model}_dataset','base','images','train',f'{name}.jpg')
               
                    im.save(im_p)
                    image_for_js = {
                                        "image_name":basename(im_p),
                                        "image_path": im_p_for_save,
                                        "label_path": txt_p_for_save,
                                        "backup_path": None,
                                        "status":{"old":None,'current':None,'uncheck':"registered"},
                                        "create_by": "auto",
                                        "reg_date": datetime.now().strftime("%Y%m%d%H%M%S"),
                                        "mod_date": datetime.now().strftime("%Y%m%d%H%M%S"),
                                        "annotation": [],
                                        'model_direction':model,
                                        'category': category,
                                        'user_id': get_user_id(username),
                                        }
                    # x, y, w, h = image_['feature_position']['x'],image_['feature_position']['y'],image_['feature_position']['w'],image_['feature_position']['h']
                    # im_h,im_w=image_['size']['height'],image_['size']['width']
                    # xywh = [x, y, w, h]
                    # pos = bnd_box_to_yolo_box(xywh, [im_h, im_w])
                    pos=image_['feature_position']
                  
                    with open(txt_p, 'a') as f:
                    
                        f.write(f"{product['class_id']} {pos['x_center']} {pos['y_center']} {pos['width']} {pos['height']}\n")
                    
                
                    anoo_box = {
                        "class_id": product['class_id'],
                        "boxes": [
                            {
                                "x_center": float(pos['x_center']),
                                "y_center": float(pos['y_center']),
                                "width": float(pos['width']),
                                "height": float(pos['height'])
                            }
                        ],
                        "status": "activate",
                    }
                    image_for_js["annotation"].append(anoo_box)
                    dataset_coll.insert_one(image_for_js)

                    
        

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


def get_images(images, product_ds, class_id=None):
    # TODO check datetime
    for idx_, im_base64 in enumerate(product_ds['image_base64']):
        im = base642PIL(im_base64['im_base64'])
        name = f"{class_id}_{product_ds['design_id']}-{str(idx_)}"
        image = {name: im}
        images.append(image)



def get_data_from_api(URL=None):
    r = requests.get(url=URL)
    try:
        data = r.json()
        return data
    except Exception as e:
        print(e)
        return None


def converse2jpg(path):
    for f in path:
        if f.endswith('png'):
            im1 = Image.open(f).convert("RGB")
            im1.save(f[:-4] + '.jpg')
            os.remove(f)


@app.route('/')
def aiimg_execute():
    return 'This is Flask Server for train Yolov5'




def bnd_box_to_yolo_box(box, img_size):
    """
    It takes a bounding box and an image size and returns the YOLO box

    :param box: the bounding box in the format (x_min, y_min, w, h)
    :param img_size: The size of the image
    :return: x_center, y_center, w, h
    """
    (x_min, y_min) = (box[0], box[1])
    (w, h) = (box[2], box[3])
    x_max = x_min + w
    y_max = y_min + h

    x_center = float((x_min + x_max)) / 2 / img_size[1]
    y_center = float((y_min + y_max)) / 2 / img_size[0]

    w = float((x_max - x_min)) / img_size[1]
    h = float((y_max - y_min)) / img_size[0]

    return x_center, y_center, w if w < 1 else 1.0, h if h < 1 else 1.0


def toImgOpenCV(img_pil):  # Converse imgPIL to imgOpenCV
    """
    Convert a PIL image to an OpenCV image by swapping the red and blue channels

    :param img_pil: The image to be converted
    :return: A numpy array
    """
    i = np.array(img_pil)  # After mapping from PIL to numpy : [R,G,B,A]
    # numpy Image Channel system: [B,G,R,A]
    red = i[:, :, 0].copy()
    i[:, :, 0] = i[:, :, 2].copy()
    i[:, :, 2] = red
    return i


def w_h_image_rotate(image):
    """
    It takes an image, converts it to RGBA, then converts it to RGB, then converts it to grayscale, then
    finds the largest contour, then returns the bounding box of that contour

    :param image: the image to be cropped
    :return: the x, y, width, and height of the image.
    """
    im = toImgOpenCV(image)
    im_to_crop = toImgOpenCV(image)

    alpha_channel = im[:, :, 3]
    rgb_channel = im[:, :, :3]
    white_background = np.ones_like(rgb_channel, dtype=np.uint8) * 255

    alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)

    base = rgb_channel.astype(np.float32) * alpha_factor
    white = white_background.astype(np.float32) * (1 - alpha_factor)
    final_im = base + white
    final_im = final_im.astype(np.uint8)

    gray = cv2.cvtColor(final_im, cv2.COLOR_BGR2GRAY)
    r1, t1 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    # t1=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    c1, h1 = cv2.findContours(t1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cnt = sorted(c1, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(cnt[0])
    return x, y, (x + w), (y + h)


def class_id(file):
    files = file.split("_")
    return str(files[0])


def design_id(file):
    files = file.split("_")  # x-x-resize
    return str(files[1]).split("-")[0]


class MergeThread(threading.Thread):
    def __init__(
        self,
        bgs,
        fgs,
        model=None,
        rotate=None,
        rotate_step=10,
        cutout_=False,
        DATA_TRANSFER=None
    ):
        super(MergeThread, self).__init__()
        self.fgs = fgs
        self.bgs = bgs
        self.rotate = rotate
        self.rotate_step = rotate_step
        self.cutout = cutout_
        self.model = model
        self.DATA_TRANSFER = DATA_TRANSFER

    def run(self):  # Get list foregrounds
        overlap_value = 10  # Overlap value
        idx = 0  # Image index
        while True:
            if len(self.fgs) == 0:
                break
            bg = random.choice(
                self.bgs
            )  # Random choice one of background in list backgrounds
            merged_image = bg.copy()
            bw, bh = merged_image.size  # Get weight height of merge image
            cur_h, cur_w, max_h, max_w = 0, 0, 0, 0
            name_=uuid.uuid4()
            txt_p = join(self.DATA_TRANSFER['DATA_ROOT'], f"{self.model}_dataset", "base", "labels", "train",f"{name_}.txt", )
            txt_p_save = (
                rf"{self.model}_dataset\base\labels\train\{name_}.txt"
            )

            im_p = join(
                self.DATA_TRANSFER['DATA_ROOT'],
                f"{self.model}_dataset",
                "base",
                "images",
                "train",
                f"{name_}.jpg",
            )
            im_p_save = (
                rf"{self.model}_dataset\base\images\train\{name_}.jpg"
            )
            image_for_js = {
                "image_name": basename(im_p),
                "image_path": im_p_save,
                "label_path": txt_p_save,
                "backup_path": None,
                "status":{"old":None,'current':None,'uncheck':"registered"},
                "create_by": "auto",
                "reg_date": datetime.now().strftime("%Y%m%d%H%M%S"),
                "mod_date": datetime.now().strftime("%Y%m%d%H%M%S"),
                "annotation": [],
                'model_direction':self.model,
                'category':self.DATA_TRANSFER['category'],
                'user_id': self.DATA_TRANSFER['user_id'],
            }
            while True:
                if len(self.fgs) == 0: 
                    break

                # Random choice foreground
                fg_dic = random.choice(self.fgs) 

                  # Get id of product
                id_ = class_id(list(fg_dic.keys())[0])
                fore_image = list(fg_dic.values())[0]
                if self.rotate is not None:
                    fore_image = fore_image.rotate(
                        random.randrange(0, int(self.rotate), self.rotate_step),
                        expand=True,
                    )
                    fore_image = fore_image.crop(w_h_image_rotate(fore_image))
                if self.cutout:
                    fore_image = cutout(fore_image, is_pil=True)
                fw, fh = fore_image.size  # Foreground size
                if fw > bw or fh > bh:
                    self.fgs.remove(fg_dic)
                    continue
                if max_h < fh:
                    max_h = fh - overlap_value
                if (cur_w + fw) >= bw:
                    cur_w = 0
                    cur_h += max_h
                if (cur_h + fh) >= bh:
                    break
                if cur_w > 0:
                    if cur_h == 0:
                        merged_image.paste(
                            fore_image, (cur_w - overlap_value, cur_h), fore_image
                        )
                        x, y = cur_w - overlap_value, cur_h
                    else:
                        merged_image.paste(
                            fore_image, (cur_w, cur_h - overlap_value), fore_image
                        )
                        x, y = cur_w, cur_h - overlap_value
                else:
                    merged_image.paste(fore_image, (cur_w, cur_h), fore_image)
                    x, y = cur_w, cur_h
                box = (x, y, fw, fh)
                yolo_box = bnd_box_to_yolo_box(
                    box, (bh, bw)
                )  # Converse Bounding Box(xywh) to Yolo format(xyxy)
                cls = int(id_)
                with open(txt_p, "a") as f:
                    f.write(
                        f"{cls} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n"
                    )

                class_id_list = [int(anno["class_id"]) for anno in image_for_js["annotation"]]
                if int(cls) in class_id_list:
                    for anoo in image_for_js["annotation"]:
                        if int(anoo["class_id"]) == int(cls):
                            position = {
                                "x_center": float(yolo_box[0]),
                                "y_center": float(yolo_box[1]),
                                "width": float(yolo_box[2]),
                                "height": float(yolo_box[3])
                            }
                            anoo["boxes"].append(position)

                            break
                else:
                    anoo_box = {
                        "class_id": int(cls),
                        "boxes": [
                            {
                                "x_center": float(yolo_box[0]),
                                "y_center": float(yolo_box[1]),
                                "width": float(yolo_box[2]),
                                "height": float(yolo_box[3])
                            }
                        ],
                        "status": "activate",
                    }
                    image_for_js["annotation"].append(anoo_box)

                # TODO : set design
                # with open(name + "-for_ds.txt", 'a') as f:
                #     f.write(f"{cls} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]} {design_id_}\n")
                cur_w += fw - overlap_value
                self.fgs.remove(fg_dic)

            # merged_image.save(name + ".png", format="png")
            merged_image_opencv = toImgOpenCV(merged_image)
            transform = A.Compose([A.RandomBrightnessContrast()])
            image = cv2.cvtColor(merged_image_opencv, cv2.COLOR_BGR2RGB)
            transformed = transform(image=image)
            transformed_image = transformed["image"]
            transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(im_p, transformed_image)
            
            if isfile(txt_p):
                with threading.Lock():
                    dataset_coll.insert_one(image_for_js)
            idx += 1


def resize(
    registing_products, min_size=0.7, max_size=1.2, step=0.1, im_num=5, DATA_ROOT=None
):
    resized_images = {}

    for product in registing_products:
        for idx, image_ in enumerate(product["images"]):
            name = f"{product['class_id']}_-{str(idx)}"
            image_path = image_["image_path"].replace("\\", "/")
            im = Image.open(join(DATA_ROOT, image_path)).convert("RGBA")

            for i in np.arange(min_size, max_size, step):
                width, height = im.size
                aspect_ratio = width / height
                new_width = width * i
                new_height = round(new_width / aspect_ratio)
                im_rs = im.resize(
                    (round(new_width), new_height), resample=Image.LANCZOS
                )
                range_ = (
                    range(1, im_num + 5, 1)
                    if round(i, 1) == 1.0
                    else range(1, im_num, 1)
                )
                for j in range_:
                    im_name = f"{name}-resize_{round(i, 1)}_{j}.jpg"
                    resized_images[im_name] = im_rs

    return resized_images


def merge_thread(
    foregrounds,
    backgrounds,
    model=None,
    rotate_=None,
    rotate_step=None,
    cutout_=False,
    DATA_TRANSFER=None
):
    dic_by_size = {}
    threads = []
    for im_name, im in foregrounds.items():
        image_names = im_name.split("-")
        sizes = str(str(image_names[2]).split("_")[1])
        if sizes not in dic_by_size.keys():
            im_ = {im_name: im}
            dic_by_size[sizes] = [im_]
        else:
            im_ = {im_name: im}
            dic_by_size[sizes].append(im_)

    for v in dic_by_size.values():
        t = MergeThread(
            bgs=backgrounds,
            fgs=v,
            model=model,
            rotate=rotate_,
            rotate_step=rotate_step,
            cutout_=cutout_,
            DATA_TRANSFER=DATA_TRANSFER
        )
        threads.append(t)
        t.start()
    for th in threads:
        th.join()


def cutout(im, is_pil=False):
    """
    It takes an image and randomly selects a square region of the image to replace with a random color

    :param im: the image to be cutout
    :param is_pil: whether the input image is a PIL image or a numpy array, defaults to False (optional)
    :return: the image with the cutout applied.
    """
    if is_pil:
        w, h = im.size
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 2
        s = random.choice(scales)
        mask_h = random.randint(1, int(h * s))  # create random masks
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        img = Image.new(
            "RGBA",
            (mask_w, mask_h),
            (random.randint(64, 191), random.randint(64, 191), random.randint(64, 191)),
        )
        im.paste(img, (xmin, ymin), img)
        return im

    else:
        h, w = im.shape[:2]
        # scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 2
        s = random.choice(scales)
        mask_h = random.randint(1, int(h * s))  # create random masks
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(4)]
        return im

# endregion

if __name__ == '__main__':
    app.run()