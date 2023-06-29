

import cv2
import numpy as np
from PIL import Image, JpegImagePlugin
import random
import threading
from os.path import join
import albumentations as A
# from pathlib import Path


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
    return str(files[1]).split('-')[0]


class MergeThread(threading.Thread):
    def __init__(self, bgs, fgs, rotate=None, rotate_step=10, cutout_=False, save_path="", file_name="merged"):
        super(MergeThread, self).__init__()
        self.fgs = fgs
        self.bgs = bgs
        self.rotate = rotate
        self.rotate_step = rotate_step
        self.cutout = cutout_
        self.save_path = save_path
        self.file_name = file_name

    def run(self):  # Get list foregrounds
        overlap_value = 10  # Overlap value
        idx = 0  # Image index
        name = "no_name"
        while True:
            if len(self.fgs) == 0:
                break
            bg = random.choice(self.bgs)  # Random choice one of background in list backgrounds
            merged_image = bg.copy()
            bw, bh = merged_image.size  # Get weight height of merge image
            cur_h, cur_w, max_h, max_w = 0, 0, 0, 0
            while True:
                if len(self.fgs) == 0:  # Break when len of list foreground == 0
                    # Break when len of list foreground == 0
                    break
                fg_dic = random.choice(self.fgs)  # Random choice foreground
                id_ = class_id(list(fg_dic.keys())[0])  # Get id of product
                design_id_ = design_id(list(fg_dic.keys())[0])  # Get design id of product
                fore_image = list(fg_dic.values())[0]
                if self.rotate is not None:
                    fore_image = fore_image.rotate(random.randrange(0, int(self.rotate), self.rotate_step), expand=True)
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
                        merged_image.paste(fore_image, (cur_w - overlap_value, cur_h), fore_image)
                        x, y = cur_w - overlap_value, cur_h
                    else:
                        merged_image.paste(fore_image, (cur_w, cur_h - overlap_value), fore_image)
                        x, y = cur_w, cur_h - overlap_value
                else:
                    merged_image.paste(fore_image, (cur_w, cur_h), fore_image)
                    x, y = cur_w, cur_h
                box = (x, y, fw, fh)
                yolo_box = bnd_box_to_yolo_box(box, (bh, bw))  # Converse Bounding Box(xywh) to Yolo format(xyxy)
                cls = int(id_)
                name = join(self.save_path, f"{self.file_name}_{idx}")
                with open(name + ".txt", 'a') as f:
                    f.write(f"{cls} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n")

                #TODO : set design
                # with open(name + "-for_ds.txt", 'a') as f:
                #     f.write(f"{cls} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]} {design_id_}\n")
                cur_w += fw - overlap_value
                self.fgs.remove(fg_dic)

            # merged_image.save(name + ".png", format="png")
            merged_image_opencv = toImgOpenCV(merged_image)
            transform = A.Compose([
                A.RandomBrightnessContrast(),
                A.Blur()])
            image = cv2.cvtColor(merged_image_opencv, cv2.COLOR_BGR2RGB)
            transformed = transform(image=image)
            transformed_image = transformed["image"]
            transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(name + ".jpg", transformed_image)
            idx += 1


def resize(registing_products, min_size=0.7, max_size=1.2, step=0.1, im_num=5,DATA_ROOT=None):
    """
    It takes a dictionary of images, resizes them, and returns a dictionary of resized images
    
    :param images: a dictionary of images
    :param min_size: the minimum size of the image to resize to
    :param max_size: the maximum size of the image
    :param step: the step size for the resize factor
    :param im_num: number of images to generate for each resize, defaults to 5 (optional)
    :return: A dictionary of images with the key being the image name and the value being the image.
    """
    resized_images = {}
    registing_ids=[product['class_id'] for product in registing_products]

    for product in registing_products:
        for design in product['design']:
            for idx,image_ in enumerate(design['images']):
                name = f"{product['class_id']}_{design['design_id']}-{str(idx)}"
                print(DATA_ROOT)
                print(image_['image_path'])
                image_path =image_['image_path'].replace('\\','/')
                im= Image.open(join(DATA_ROOT,image_path)).convert('RGBA')

                for i in np.arange(min_size, max_size, step):
                    width, height = im.size
                    aspect_ratio = width / height
                    new_width = width * i
                    new_height = round(new_width / aspect_ratio)
                    im_rs = im.resize((round(new_width), new_height), resample=Image.LANCZOS)
                    range_ = range(1, im_num + 5, 1) if round(i, 1) == 1.0 else range(1, im_num, 1)
                    for j in range_:
                        im_name = f"{name}-resize_{round(i, 1)}_{j}.jpg"
                        resized_images[im_name] = im_rs

    return resized_images


def merge_thread(foregrounds, backgrounds, p_save, rotate_=None, rotate_step=None, cutout_=False, name=None):
    """
    It takes a dictionary of foregrounds and a dictionary of backgrounds, and creates a thread for each
    foreground size, and then merges the foregrounds of that size with the backgrounds
    
    :param foregrounds: a dictionary of foreground images
    :param backgrounds: a dictionary of background images
    :param p_save: the path to save the merged images
    :param rotate_: if you want to rotate the foregrounds, you can pass a list of angles to rotate by
    :param rotate_step: step to rotate
    :param cutout_: if True, it will cutout the foreground image from the background image, defaults to
    False (optional)
    :param name: the name of the merged images, defaults to merged (optional)
    """
    dic_by_size = {}
    threads = []
    for im_name, im in foregrounds.items():
        image_names = im_name.split("-")
        sizes = str(str(image_names[2]).split('_')[1])
        if sizes not in dic_by_size.keys():
            im_ = {im_name: im}
            dic_by_size[sizes] = [im_]
        else:
            im_ = {im_name: im}
            dic_by_size[sizes].append(im_)

    for k, v in dic_by_size.items():
        t = MergeThread(bgs=backgrounds,
                        fgs=v,
                        rotate=rotate_,
                        rotate_step=rotate_step,
                        cutout_=cutout_,
                        save_path=p_save,
                        file_name=f"{name}_{k}")
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
        img = Image.new("RGBA", (mask_w, mask_h),
                        (random.randint(64, 191), random.randint(64, 191), random.randint(64, 191)))
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
