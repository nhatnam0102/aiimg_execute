import math
import os
import shutil

import cv2
import numpy as np
from PIL import Image, JpegImagePlugin
import random
import threading
from upc.common import images_process as u
from datetime import datetime
from os.path import join
from os.path import basename
import logging

import csv

from rembg import remove
import onnxruntime
import albumentations as A
from pathlib import Path

ROOT = r"\\192.168.0.241\nam\yakult_project"
IMAGE_PROCESS_PATH = r"\\192.168.0.241\nam\yakult_project\images_processed"

FILE = Path(__file__).resolve()
BASEDIR = FILE.parents[0]


def set_logging(log_name,
                file_log_name,
                formatter="%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    # Format with pathname and lineno
    # formatter=Formatter('%(asctime)s - %(levelname)s - [in %(pathname)s:%(lineno)d] %(message)s]')

    # Save log to file
    # f_handler = FileHandler(join(os.getcwd(), "data", "log", file_log_name))
    # f_handler.setLevel(logging.INFO)
    # f_handler.setFormatter(Formatter(formatter))
    # logger.addHandler(f_handler)

    # Show log in terminal(for DEBUG)
    s_handler = logging.StreamHandler()
    s_handler.setLevel(logging.INFO)

    logger.addHandler(s_handler)

    return logger


# LOGGER
LOGGER = set_logging("AI DETECT APP", "mylog.log")


def get_all_file(path_dir):
    """
    Get all file in directory
    :param path_dir: Path of directory
    :return: list of file, list of directory in directory
    """
    file_list, dir_list = [], []
    for rdir, subdir, files in os.walk(path_dir):
        file_list.extend([os.path.join(rdir, f) for f in files])
        dir_list.extend([os.path.join(rdir, d) for d in subdir])
    return file_list, dir_list


def remove_background(image, save_path=None):
    """
    Remove background of image.

    :param image: path of image
    :param save_path: save path
    :return: -Write removed image in save_path if save path is not None.
     -Return removed image(OpenCV)
    """
    in_ = cv2.imread(image,cv2.IMREAD_UNCHANGED)
    out_ = remove(in_)
    if save_path is not None:
        cv2.imwrite(save_path, out_)
    else:
        return out_


def toImgOpenCV(img_pil):  # Converse imgPIL to imgOpenCV
    i = np.array(img_pil)  # After mapping from PIL to numpy : [R,G,B,A]
    # numpy Image Channel system: [B,G,R,A]
    red = i[:, :, 0].copy()
    i[:, :, 0] = i[:, :, 2].copy()
    i[:, :, 2] = red
    return i


# CAN'T CONVERSE OPENCV B,G,R,A TO PIL R,G,B.,A
def toImgPIL(img_opencv): return Image.fromarray(cv2.cvtColor(img_opencv, cv2.COLOR_BGRA2RGBA))


def w_h_image_rotate(image):
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


def cut_bg_image(im):
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
    crop = im[y:y + h, x:x + w]
    return crop


def cut_from_removed_background(image, save_path=None, is_pil=False):
    if is_pil:
        im = image
        im_to_crop = im.copy()

    else:

        im = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        im_to_crop = im.copy()

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
    crop = im_to_crop[y:y + h, x:x + w]
    cv2.imwrite(save_path, crop)


def class_id(file):
    files = file.split("_")
    return str(files[0])


def yolo_box_to_rec_box(box, img_size):
    x, y, w, h = box
    x1 = int((x - w / 2) * img_size[1])
    w1 = int((x + w / 2) * img_size[1])
    y1 = int((y - h / 2) * img_size[0])
    h1 = int((y + h / 2) * img_size[0])
    if x1 < 0:
        x1 = 0
    if w1 > img_size[1] - 1:
        w1 = img_size[1] - 1
    if y1 < 0:
        y1 = 0
    if h1 > img_size[0] - 1:
        h1 = img_size[0] - 1
    return x1, y1, w1 - x1, h1 - y1


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


def merge(path_background, path_foreground, annotation_num, save_path, rotate=False, filename="merged_2"):
    """Paste foreground to background """

    backgrounds, _ = get_all_file(path_background)  # Get list backgrounds
    foregrounds, _ = get_all_file(path_foreground)  # Get list foregrounds
    dictionary = {}  # ex: {"1": [1,0]}  => [1,0] : 1 annotation value, 0 rotate degree value
    overlap_value = 10  # Overlap value
    idx = 0  # Image index
    name = "no_name"
    while True:
        if len(foregrounds) == 0:  # Break when len of list foreground == 0
            break

        # Random choice one of background in list backgrounds with 4 channel
        bg = Image.open(random.choice(backgrounds)).convert("RGBA")
        merged_image = bg.copy()
        bw, bh = merged_image.size  # Get weight height of merge image
        cur_h, cur_w, max_h, max_w = 0, 0, 0, 0
        is_write = False
        while True:
            if len(foregrounds) == 0:
                break
            fg = random.choice(foregrounds)  # Random choice foreground
            id_ = class_id(basename(fg))
            if id_ not in dictionary:
                dictionary[id_] = [1, 0]
            else:
                if dictionary[id_][0] > annotation_num:
                    print(f"{id_} finnish")
                    foregrounds.remove(fg)
                    continue
            fore_image = Image.open(fg).convert("RGBA")  # Read foreground image with 4 channels
            if rotate:
                if dictionary[id_][1] > 359:
                    dictionary[id_][1] = 0
                dictionary[id_][1] += 20
                fore_image = fore_image.rotate(dictionary[id_][1], expand=True)
                fore_image = fore_image.crop(w_h_image_rotate(fore_image))
            fw, fh = fore_image.size  # Background size
            if fw > bw or fh > bh:
                continue

            if max_h < fh:
                max_h = fh - overlap_value
            if (cur_w + fw) >= bw:
                cur_w = 0
                cur_h += max_h
            if (cur_h + fh) >= bh:
                break
            x, y = 0, 0
            try:
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
            except Exception as e:
                print(e)
            box = (x, y, fw, fh)
            yolo_box = bnd_box_to_yolo_box(box, (bh, bw))  # Converse Bounding Box(xywh) to Yolo format(xyxy)
            cls = int(id_)
            name = join(save_path, f"{filename}_{idx}")
            with open(name + ".txt", 'a') as f:
                f.write(f"{cls} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n")
            is_write = True
            dictionary[id_][0] += 1
            cur_w += fw - overlap_value

        if is_write:
            merged_image.save(name + ".png", format="png")
        idx += 1


def get_frames(src: str, save_, value=10):
    if src.endswith("png"):
        folder_id = join(save_, class_id(basename(src)))
        os.makedirs(folder_id, exist_ok=True)
        old, new = src, join(folder_id, basename(src))
        shutil.copy(old, new)
    elif src.endswith("mp4") or src.endswith("avi"):
        cap = cv2.VideoCapture(src)
        stt, index = 1, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            folder_id = join(save_, class_id(basename(src)))
            os.makedirs(folder_id, exist_ok=True)
            if index % value == 0:
                cv2.imwrite(join(folder_id, basename(src)[:-4] + f"{stt}.png"), frame)
            stt += 1
            index += 1


def count_annotation(path_txt, file_classes_name, write_to_csv=False):
    annotation_dic = {}
    f_name = open(file_classes_name, 'r')
    d_name = f_name.readlines()
    f_name.close()
    names = []
    for d in d_name:
        names.append(d.rstrip('\n'))

    files, _ = get_all_file(path_txt)
    for txt in files:
        if txt.endswith('txt'):
            f1 = open(txt, 'r')
            data = f1.readlines()
            f1.close()
            for dt in data:
                cls, x, y, w, h = map(float, dt.split(' '))
                if cls in annotation_dic.keys():
                    annotation_dic[cls] += 1
                else:
                    annotation_dic[cls] = 1

        for index, id_cls in enumerate(names):
            if index not in annotation_dic.keys():
                annotation_dic[index] = 0

        if write_to_csv:
            header = names
            data = []
            for k, v in sorted(annotation_dic.items()):
                data.append(v)

            with open('/test.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerow(data)
    print(annotation_dic)


def remove_background_thread(src, save_):
    save_f = join(save_, class_id(basename(src)))
    os.makedirs(save_f, exist_ok=True)
    remove_background(src, join(save_f, basename(src)))


def contras_brightness(src, save_light, save_dark, plus_alpha=1.6, minus_alpha=0.8):
    original_images, _ = get_all_file(src)
    alpha = 1.5
    for image in original_images:
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        plus_contras, minus_contrast = cv2.convertScaleAbs(img, alpha, plus_alpha), cv2.convertScaleAbs(img, alpha,
                                                                                                        minus_alpha)
        cv2.imwrite(join(save_light, basename(image)[:-4] + "_light.png"), plus_contras)
        cv2.imwrite(join(save_dark, basename(image)[:-4] + "_dark.png"), minus_contrast)


def affine_image(path, save_path):
    # list_file, _ = get_all_file(path)
    # for img in list_file:
    #     if img.endswith('png'):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    h, w = im.shape[:2]
    if w < h:
        pts1, pts2 = np.float32([[0, 0], [w, 0], [w / 2, h]]), np.float32([[0, h / 2], [w, h / 2], [w / 2, h]])
        M = cv2.getAffineTransform(pts1, pts2)
        img_affine = cv2.warpAffine(im, M, (w, h))
        save = join(save_path, str(basename(path)[:-4]) + '_affine.png')
        crop = img_affine[0 + int(h / 2):0 + h, 0:0 + w]
        cv2.imwrite(save, crop)
    else:
        pts1, pts2 = np.float32([[0, h / 2], [w, 0], [w, h]]), np.float32(
            [[0, h / 2], [w / 2, 0], [w / 2, h / 2]])
        M = cv2.getAffineTransform(pts1, pts2)
        img_affine = cv2.warpAffine(im, M, (w, h))
        save = join(save_path, str(basename(path)[:-4]) + '_affine.png')
        crop = img_affine[0:0 + h, 0:0 + int(w / 2)]
        cv2.imwrite(save, crop)


def affine_image_2(path, save_path):
    list_file, _ = get_all_file(path)
    for img in list_file:
        if img.endswith('png'):
            im = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            h, w = im.shape[:2]
            if h < w:
                x = random.randint(30, 50)  # for w
                z = x - 20
                pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])  # original
                pts1 = np.float32([[0, 0], [w - x, 0 + z], [0, h], [w - x, h - z]])  # affine right
                pts2 = np.float32([[0 + x, 0 + z], [w, 0], [0 + x, h - z], [w, h]])  # affine left
                M, M1 = cv2.getPerspectiveTransform(pts, pts1), cv2.getPerspectiveTransform(pts, pts2)
                img_affine_r, img_affine_l = cv2.warpPerspective(im, M, (w, h)), cv2.warpPerspective(im, M1, (w, h))
                save1, save2 = join(save_path, str(basename(img)[:-4]) + '-af_r.png'), join(save_path,
                                                                                            str(basename(img)[
                                                                                                :-4]) + '-af_l.png')
                crop1, crop2 = img_affine_r[0:0 + h, 0:0 + w - x], img_affine_l[0:0 + h, 0 + x:0 + w]
                cv2.imwrite(save1, crop1)  # Save affine right
                cv2.imwrite(save2, crop2)  # Save affine left


def affine_5_degrees(fg, bg, save, pers_size=0, fit_box=(0, 0), fit_image=(0, 0), rotate=False):
    # Perspective
    im_fg = cv2.imread(fg, cv2.IMREAD_UNCHANGED)  # foreground

    im_bg = cv2.imread(bg)  # background
    im_bg_draw = im_bg.copy()
    fg_h, fg_w = im_fg.shape[:2]
    bg_h, bg_w = im_bg.shape[:2]

    if rotate:
        # fg_size = (fg_w, fg_h)
        # dst_mat = np.zeros((fg_h, fg_w, 4), np.uint8)
        # angle = random.randint(50, 359, )
        angle = 180
        scale = 1
        M = cv2.getRotationMatrix2D(center=(fg_w // 2, fg_h // 2), angle=-angle, scale=scale)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((fg_h * sin) + (fg_w * cos))
        nH = int((fg_h * cos) + (fg_w * sin))

        M[0, 2] += (nW / 2) - (fg_w // 2)
        M[1, 2] += (nH / 2) - (fg_h // 2)
        im_fg = cv2.warpAffine(im_fg, M, (nW, nH))
        # im_fg = cv2.warpAffine(im_fg, M, fg_size, dst_mat,
        #                        flags=cv2.INTER_LINEAR,
        #                        borderMode=cv2.BORDER_TRANSPARENT)
    fg_h, fg_w = im_fg.shape[:2]
    num_w, num_h = (bg_w // fg_w), (bg_h // fg_h)  # number of foregrounds in a row,col
    cur_w, cur_h = 0, 0
    size_w = pers_size // num_h
    for y0 in range(1, num_h + 1):
        if cur_h >= bg_h:
            break
        x = pers_size // 2
        for x0 in range(1, num_w + 1):
            if cur_w > bg_w:
                continue

            pts = np.float32([[0, 0], [fg_w, 0], [0, fg_h], [fg_w, fg_h]])  # original
            if x0 < num_w // 2:
                x -= size_w // 2
                pts1 = np.float32([[0, 0 + x], [fg_w, 0], [0, fg_h - x], [fg_w, fg_h]])
            elif x0 == num_w // 2:
                x = 0
                pts1 = np.float32([[0, 0], [fg_w, 0], [0, fg_h], [fg_w, fg_h]])
            else:
                x += size_w // 2
                pts1 = np.float32([[0, 0], [fg_w, 0 + x], [0, fg_h], [fg_w, fg_h - x]])

            M = cv2.getPerspectiveTransform(pts, pts1)
            img_affine = cv2.warpPerspective(im_fg, M, (fg_w, fg_h))

            im_bg = paste_by_opencv(im_bg, img_affine, cur_w, cur_h)
            im_bg_draw = paste_by_opencv(im_bg_draw, img_affine, cur_w, cur_h)
            box = (cur_w + fit_box[0], cur_h + fit_box[1], fg_w - fit_box[0] * 2, fg_h - fit_box[1] * 2)
            cv2.rectangle(im_bg_draw, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 2)
            yolo_box = bnd_box_to_yolo_box(box, (bg_h, bg_w))  # Converse Bounding Box(xywh) to Yolo format(xyxy)

            id = basename(fg).split("_")[0]

            with open(join(save, "merged1-" + basename(fg)[:-4] + ".txt"), 'a') as f:
                f.write(f"{id} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n")
            cur_w += fg_w - fit_image[0]

        cur_h += fg_h - fit_image[1]
        cur_w = 0
    cv2.imwrite(join(save, "merged1-" + basename(fg)), im_bg)

    return im_bg, im_bg_draw


def draw_put_text(draw_img, x1, y1, x2, y2, conf, cls, point=False, not_show_conf=False, color_=None):
    if color_ is not None:
        color = color_
    else:
        if int(conf * 100) >= 80:
            color = (0, 0, 255)  # BLUE
        elif int(conf * 100) >= 50:
            color = (0, 255, 0)  # GREEN
        else:
            color = (255, 0, 0)  # RED
    if point:
        # over = draw_img.copy()
        # cv2.circle(over, (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2)), 28, color, thickness=-1,
        #            lineType=cv2.LINE_4, shift=0)
        # alpha = 0.4
        #
        # draw_img = cv2.addWeighted(over, alpha, draw_img, 1 - alpha, 0)

        # cv2.circle(draw_img, (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2)), 30, color, thickness=5,
        #            lineType=cv2.LINE_4,
        #            shift=0)
        cv2.ellipse(draw_img, center=(int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2)),
                    axes=(40, 40),
                    angle=270,
                    startAngle=0,
                    endAngle=360,
                    color=(128, 128, 128),
                    thickness=7,
                    lineType=cv2.LINE_4,
                    shift=0)
        cv2.ellipse(draw_img, center=(int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2)),
                    axes=(40, 40),
                    angle=270,
                    startAngle=0,
                    endAngle=int(conf * 360),
                    color=color,
                    thickness=7,
                    lineType=cv2.LINE_4,
                    shift=0)
        cv2.putText(draw_img, f"{cls}",
                    (int(x1 + (x2 - x1) / 2) - 20, int(y1 + (y2 - y1) / 2) + 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(draw_img, f"{cls}",
                    (int(x1 + (x2 - x1) / 2) - 20, int(y1 + (y2 - y1) / 2) + 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1, color,
                    2, cv2.LINE_AA)
    else:
        cv2.rectangle(draw_img, (x1, y1),
                      (x2, y2), color, 2)
        cv2.putText(draw_img, f"{cls} {int(conf * 100)}%", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(draw_img, f"{cls} {int(conf * 100)}%", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_COMPLEX, 1, color,
                    3, cv2.LINE_AA)

        # z = 30
        # x = 10
        # cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # if (x2 - x1) < (y2 - y1):
        #     z = (cx - x1) // 2
        #     x = z // 3
        # else:
        #     z = (cy - y1) // 2
        #     x = z // 3
        #
        # thickness = 2
        # if not_show_conf:
        #     text = f"{cls}"
        # else:
        #     text = f"{cls} {int(conf * 100)}%"
        # if cls >= 10:
        #     text_pos = (int(cx - z // 2), int(cy + z // 2))
        # else:
        #     text_pos = (int(cx - z // 3), int(cy + z // 3))
        #
        # cv2.putText(draw_img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (255, 255, 255), 7, cv2.LINE_AA)
        # cv2.putText(draw_img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color,
        #             3, cv2.LINE_AA)
        # color1 = (255, 255, 255)
        # color = (0, 255, 0)
        # thickness_border = 3
        # pts1 = np.array([[cx - z, cy - z + x], [cx - z, cy - z], [cx - z + x, cy - z]], np.int32)
        # pts1 = pts1.reshape((-1, 1, 2))
        # draw_img = cv2.polylines(draw_img, pts=[pts1], isClosed=False, color=color1,
        #                          thickness=thickness + thickness_border)
        # draw_img = cv2.polylines(draw_img, pts=[pts1], isClosed=False, color=color, thickness=thickness)
        #
        # pts1 = np.array([[cx + z, cy - z + x], [cx + z, cy - z], [cx + z - x, cy - z]], np.int32)
        # pts1 = pts1.reshape((-1, 1, 2))
        # draw_img = cv2.polylines(draw_img, pts=[pts1], isClosed=False, color=color1,
        #                          thickness=thickness + thickness_border)
        # draw_img = cv2.polylines(draw_img, pts=[pts1], isClosed=False, color=color, thickness=thickness)
        #
        # pts1 = np.array([[cx - z, cy + z - x], [cx - z, cy + z], [cx - z + x, cy + z]], np.int32)
        # pts1 = pts1.reshape((-1, 1, 2))
        # draw_img = cv2.polylines(draw_img, pts=[pts1], isClosed=False, color=color1,
        #                          thickness=thickness + thickness_border)
        # draw_img = cv2.polylines(draw_img, pts=[pts1], isClosed=False, color=color, thickness=thickness)
        #
        # pts1 = np.array([[cx + z, cy + z - x], [cx + z, cy + z], [cx + z - x, cy + z]], np.int32)
        # pts1 = pts1.reshape((-1, 1, 2))
        # draw_img = cv2.polylines(draw_img, pts=[pts1], isClosed=False, color=color1,
        #                          thickness=thickness + thickness_border)
        # draw_img = cv2.polylines(draw_img, pts=[pts1], isClosed=False, color=color, thickness=thickness)

    return draw_img


def cross_single_by_opencv(fg, bg, save, fit_box=(0, 0), fit_image=(0, 0), angle=0, mod_case=False):
    boxes = []
    im_fg = cv2.imread(fg, cv2.IMREAD_UNCHANGED)  # foreground
    im_bg = cv2.imread(bg)  # background
    im_bg_draw = im_bg.copy()
    fg_h, fg_w = im_fg.shape[:2]
    bg_h, bg_w = im_bg.shape[:2]
    scale = 1
    M = cv2.getRotationMatrix2D(center=(fg_w // 2, fg_h // 2), angle=angle, scale=scale)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image

    nW = int((fg_h * sin) + (fg_w * cos))
    nH = int((fg_h * cos) + (fg_w * sin))
    h_test = int((fg_w * sin))

    print(f"fdf  {h_test}")
    M[0, 2] += (nW / 2) - (fg_w // 2)
    M[1, 2] += (nH / 2) - (fg_h // 2)
    im_fg = cv2.warpAffine(im_fg, M, (nW, nH))

    cur_w, cur_h = 0, 0
    for y0 in range(0, 1):
        if cur_h >= bg_h:
            break
        for x0 in range(0, 5):
            if cur_w + fg_w > bg_w:
                break
            # im_bg = paste_by_opencv(im_bg, im_fg, cur_w, cur_h)
            print(cur_w)
            im_bg = paste_by_opencv(im_bg, im_fg, cur_w, cur_h)
            im_bg_draw = paste_by_opencv(im_bg_draw, im_fg, cur_w, cur_h)
            # Test
            box = (cur_w + fit_box[0] + h_test // 2,
                   cur_h + fit_box[1] + h_test // 2,
                   nW - fit_box[0] * 2 - h_test,
                   nH - fit_box[1] * 2 - h_test)

            boxes.append(box)

            cur_w += fg_w - fit_image[0]

            if mod_case:
                if angle < 0:
                    cur_w -= h_test // 2
                    cur_h += h_test - fit_image[0] // 2  # Test
                else:  # Test
                    cur_w += h_test // 2
                    cur_h += h_test // 2

        cur_h += fg_h - fit_image[1]
    for box in boxes:
        yolo_box = bnd_box_to_yolo_box(box, (bg_h, bg_w))  # Converse Bounding Box(xywh) to Yolo format(xyxy)
        id = basename(fg).split("_")[0]

        with open(join(save, basename(fg)[:-4] + ".txt"), 'a') as f:
            f.write(f"{id} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n")
        im_bg_draw = draw_put_text(im_bg_draw,
                                   box[0], box[1], box[0] + box[2], box[1] + box[3],
                                   1,
                                   0,
                                   point=False,
                                   not_show_conf=True,
                                   color_=(random.randint(61, 193), random.randint(61, 193), random.randint(61, 193)))

    cv2.imwrite(join(save, basename(fg)), im_bg)

    return im_bg, im_bg_draw


def rotate_image(image, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    height, width = image.shape[:2]  # image shape has 3 dimensions
    image_center = (width / 2,
                    height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origin) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def cross_by_pil(fit_box=(0, 0), fit_image=(0, 0), angle=-30):
    bg = Image.open("./transparent.png").convert("RGBA")
    fg = Image.open("./c.png").convert("RGBA")
    boxes = []
    merged_image = bg.copy()
    bw, bh = merged_image.size  # Get weight height of merge image
    fw, fh = fg.size

    cur_w, cur_h = 0, 0
    a0, a1 = 0, 0
    for y0 in range(0, 4):
        if cur_h + fh >= bh:
            break
        for x0 in range(0, 3):
            if cur_w + fw > bw:
                break
            merged_image.paste(fg, (cur_w, cur_h), fg)
            box = (cur_w + fit_box[0], cur_h + fit_box[1], fw - fit_box[0] * 2, fh - fit_box[1] * 2)
            boxes.append(box)
            cur_w += fw - fit_image[0]
            a0 = cur_w

        cur_w = 0
        a1 = cur_h
        cur_h += fh - fit_image[1]
    merged_image = toImgOpenCV(merged_image)
    merged_image = merged_image[:a1, :a0]
    print(merged_image.shape[:2])
    rotate_angle = angle * np.pi / 180
    rot_M = np.array([[np.cos(rotate_angle), -np.sin(rotate_angle)], [np.sin(rotate_angle), np.cos(rotate_angle)]])
    merged_image = rotate_image(merged_image, -45)
    cv2.imwrite("mer2.png", merged_image)
    new_height, new_width = merged_image.shape[:2]
    print(f"{new_width}x {new_height}")
    new_bbox = []
    H, W = fh, fw
    im_bg = cv2.imread("./bg.png")  # background
    # im_bg=cv2.resize(im_bg,(0,0),fx=merged_image.shape[0]/1080,fy=merged_image.shape[0]/1080)
    im_bg = paste_by_opencv(im_bg, merged_image, 0, 0)
    for box in boxes:
        (center_x, center_y, bbox_width, bbox_height) = box
        # shift the origin to the center of the image.
        upper_left_corner_shift = (center_x // 2 - W / 2, -H / 2 + center_y // 2)
        upper_right_corner_shift = (bbox_width - W / 2, -H / 2 + center_y // 2)
        lower_left_corner_shift = (center_x // 2 - W / 2, -H / 2 + bbox_height)
        lower_right_corner_shift = (bbox_width - W / 2, -H / 2 + bbox_height)

        new_lower_right_corner = [-1, -1]
        new_upper_left_corner = []

        for i in (upper_left_corner_shift, upper_right_corner_shift, lower_left_corner_shift,
                  lower_right_corner_shift):
            new_coords = np.matmul(rot_M, np.array((i[0], -i[1])))
            x_prime, y_prime = new_width / 2 + new_coords[0], new_height / 2 - new_coords[1]
            if new_lower_right_corner[0] < x_prime:
                new_lower_right_corner[0] = x_prime
            if new_lower_right_corner[1] < y_prime:
                new_lower_right_corner[1] = y_prime

            if len(new_upper_left_corner) > 0:
                if new_upper_left_corner[0] > x_prime:
                    new_upper_left_corner[0] = x_prime
                if new_upper_left_corner[1] > y_prime:
                    new_upper_left_corner[1] = y_prime
            else:
                new_upper_left_corner.append(x_prime)
                new_upper_left_corner.append(y_prime)
            #             print(x_prime, y_prime)

            new_bbox.append([0, new_upper_left_corner[0], new_upper_left_corner[1],
                             new_lower_right_corner[0], new_lower_right_corner[1]])
    print(len(new_bbox))
    for box in new_bbox:
        bb = yolo_box_to_rec_box((box[1], box[2], box[3], box[4]), im_bg.shape[:2])
        im_bg = draw_put_text(im_bg,
                              bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3],
                              1,
                              0,
                              point=False,
                              not_show_conf=True,
                              color_=(random.randint(61, 193), random.randint(61, 193), random.randint(61, 193)))
    cv2.imwrite("aa.png", im_bg)


def paste_by_opencv(bg, fg, x_offset=0, y_offset=0):
    y1, y2 = y_offset, y_offset + fg.shape[0]
    x1, x2 = x_offset, x_offset + fg.shape[1]
    alpha_fg = fg[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_fg
    for c in range(0, 3):
        bg[y1:y2, x1:x2, c] = (alpha_fg * fg[:, :, c] + alpha_bg * bg[y1:y2, x1:x2, c])
    return bg


def in_line(path, save_path, value=30):
    files, _ = get_all_file(path)
    for img in files:
        image = cv2.imread(img)
        h, w = image.shape[:2]
        image_in_line = image[0 + value:0 + h - value, 0 + value:0 + w - value]
        save = join(save_path, str(basename(img)[:-4]) + '-inline.png')
        cv2.imwrite(save, image_in_line)


def add_plastic(path_wrap, path_foreground, save_path):
    wraps, _ = get_all_file(path_wrap)  # Get list wrap plastic
    foregrounds, _ = get_all_file(path_foreground)  # Get list foregrounds
    idx = 0  # Image index
    while True:
        if len(foregrounds) == 0:
            break
        bg_path = random.choice(foregrounds)
        print(bg_path)
        bg = Image.open(bg_path).convert("RGBA")
        merged_image = bg.copy()
        bw, bh = merged_image.size  # Get weight height of merge image

        wrap = random.choice(wraps)
        wrap_img = Image.open(wrap).convert("RGBA")
        wrap_w, wrap_h = wrap_img.size

        ratio = bw / wrap_w
        wrap_img = wrap_img.resize((int(wrap_w * ratio), int(wrap_h * ratio)))
        print(f"{wrap_img.size} =>> {bg.size}")

        merged_image.paste(wrap_img, (0, 0), wrap_img)

        name = join(save_path, f"{basename(bg_path)[:-4]}-wrap_{idx}")

        foregrounds.remove(bg_path)

        merged_image.save(name + ".png", format="png")
        idx += 1


def create_dataset(dataset_path, images_path, bg_dir):
    # Init Dataset Directory
    path_img_train, path_img_valid = join(dataset_path, 'images', 'train'), join(dataset_path, 'images', 'valid')
    path_lbl_train, path_lbl_valid = join(dataset_path, 'labels', 'train'), join(dataset_path, 'labels', 'valid')
    os.makedirs(path_img_train, exist_ok=True)
    os.makedirs(path_img_valid, exist_ok=True)
    os.makedirs(path_lbl_train, exist_ok=True)
    os.makedirs(path_lbl_valid, exist_ok=True)

    images, labels = [], []
    for file in os.listdir(images_path):
        images.append(file) if file.endswith('png') else labels.append(file)

    num_of_valid = len(images) // 5
    print(f'Len of images : {len(images)}\nLen of labels : {len(labels)}\nLen valid : {num_of_valid}')

    random_images = []
    # for i in range(0, num_of_valid, 1):
    #     img_ran = random.choice(images)
    #     random_images.append(img_ran[:-4])
    #     images.remove(img_ran)

    for img in os.listdir(images_path):
        # Add valid image and label to Dataset Directory
        if img[:-4] in random_images:
            shutil.copy(join(images_path, img), join(path_img_valid, img)) if img.endswith(
                'png') else shutil.copy(join(images_path, img), join(path_lbl_valid, img))

        # Add train image and label to Dataset Directory
        else:
            shutil.copy(join(images_path, img), join(path_img_train, img)) if img.endswith(
                'png') else shutil.copy(join(images_path, img), join(path_lbl_train, img))

    # Add background image to train dataset
    for img in os.listdir(bg_dir):
        shutil.copy(join(bg_dir, img), join(path_img_train, img))


def check_annotation(path_check_labels, path_check_train, path_save):
    files, _ = get_all_file(path_check_labels)
    index = 0

    for f in files:
        if basename(f) == "classes.txt":
            continue
        elif f.endswith("txt"):
            if basename(f)[:-4] + ".png" in os.listdir(path_check_train):
                image = cv2.imread(join(path_check_train, basename(f)[:-4] + ".png"))
                with open(f, "r", encoding="UTF-8") as txt:
                    for line in txt:
                        texts = line.split(" ")
                        save_f = join(path_save, texts[0])
                        os.makedirs(save_f, exist_ok=True)
                        box = (float(texts[1]), float(texts[2]), float(texts[3]), float(texts[4]))
                        x, y, w, h = yolo_box_to_rec_box(box, image.shape[:2])
                        crop = image[y:y + h, x:x + w]
                        index += 1
                        save_name = basename(f)[:-4] + f"_{index}.png"
                        cv2.imwrite(join(save_f, save_name), crop)
            elif basename(f)[:-4] + ".jpg" in os.listdir(path_check_train):
                image = cv2.imread(join(path_check_train, basename(f)[:-4] + ".jpg"))
                with open(f, "r", encoding="UTF-8") as txt:
                    for line in txt:
                        texts = line.split(" ")
                        save_f = join(path_save, texts[0])
                        os.makedirs(save_f, exist_ok=True)
                        box = (float(texts[1]), float(texts[2]), float(texts[3]), float(texts[4]))
                        x, y, w, h = yolo_box_to_rec_box(box, image.shape[:2])
                        crop = image[y:y + h, x:x + w]
                        index += 1
                        save_name = basename(f)[:-4] + f"_{index}.png"
                        cv2.imwrite(join(save_f, save_name), crop)
            else:
                continue


class MergeThread(threading.Thread):
    def __init__(self, bgs, fgs, rotate=None,rotate_step=10, cutout_=False, save_path="", file_name="merged"):
        super(MergeThread, self).__init__()
        self.fgs = fgs
        self.bgs = bgs
        self.rotate = rotate
        self.rotate_step=rotate_step
        self.cutout = cutout_
        self.save_path = save_path
        self.file_name = file_name
    

    def run(self):  # Get list foregrounds
        overlap_value = 10  # Overlap value
        idx = 0  # Image index
        name = "no_name"
        while True:
            if len(self.fgs) == 0:  # Break when len of list foreground == 0
                break
            bg_path = random.choice(self.bgs)  # Random choice one of background in list backgrounds
            bg = Image.open(bg_path).convert("RGBA")  # Read bg by Image PIL with 4 channels
            print(f"Processing {basename(bg_path)}")
            merged_image = bg.copy()
            bw, bh = merged_image.size  # Get weight height of merge image
            cur_h, cur_w, max_h, max_w = 0, 0, 0, 0
            while True:
                if len(self.fgs) == 0:  # Break when len of list foreground == 0
                    break
                fg = random.choice(self.fgs)  # Random choice foreground
                fore_image = Image.open(fg).convert("RGBA")  # Read foreground image with 4 channels
                id_ = class_id(basename(fg))  # Get id of product
                if self.rotate is not None:
                    fore_image = fore_image.rotate(random.randrange(0, int(self.rotate), rotate_step), expand=True)
                    fore_image = fore_image.crop(w_h_image_rotate(fore_image))
                if self.cutout:
                    fore_image = cutout(fore_image, is_pil=True)
                fw, fh = fore_image.size  # Foreground size
                if fw > bw or fh > bh:
                    continue
                if max_h < fh:
                    max_h = fh - overlap_value
                if (cur_w + fw) >= bw:
                    cur_w = 0
                    cur_h += max_h
                if (cur_h + fh) >= bh:
                    break
                x, y = 0, 0
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
                cur_w += fw - overlap_value
                self.fgs.remove(fg)

            # merged_image.save(name + ".png", format="png")
            merged_image_opencv = toImgOpenCV(merged_image)
            transform = A.Compose([
                A.RandomBrightnessContrast(),
                A.Blur()])
            image = cv2.cvtColor(merged_image_opencv, cv2.COLOR_BGR2RGB)
            transformed = transform(image=image)
            transformed_image = transformed["image"]
            transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(name + ".png", transformed_image)
            idx += 1


def merge_thread(p_foregrounds, p_backgrounds, p_save, rotate_=None, cutout_=False, name="merged"):
    lst_foregrounds, _ = get_all_file(p_foregrounds)
    lst_backgrounds, _ = get_all_file(p_backgrounds)
    dic_by_size = {}
    threads = []
    for f in lst_foregrounds:
        print(f)
        image_names = basename(f).split("-")
        sizes = str(str(image_names[2]).split('_')[1])
        if sizes not in dic_by_size.keys():
            dic_by_size[sizes] = [f]
        else:
            dic_by_size[sizes].append(f)

    for k, v in dic_by_size.items():
        t = MergeThread(bgs=lst_backgrounds,
                        fgs=v,
                        rotate=rotate_,
                        cutout_=cutout_,
                        save_path=p_save,
                        file_name=f"{name}_{k}")
        threads.append(t)
        t.start()
    for th in threads:
        th.join()


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        # cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed
        return cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)


def change_id(dict_change, p_change):
    lst_txt, _ = get_all_file(p_change)
    for txt in lst_txt:
        if txt.endswith("txt"):
            with open(txt, "r") as f_:
                data = [line.strip() for line in f_]
            with open(txt, "w+") as f1:
                for text in data:
                    texts = text.split(" ")

                    if texts[0] in dict_change.keys():
                        new_line = f"{dict_change[str(texts[0])]} {texts[1]} {texts[2]} {texts[3]} {texts[4]}"
                        f1.writelines(new_line)
                        f1.writelines("\n")
                    else:
                        continue


def resize(original_files, save, start=0.7, stop=1.2, step=0.1, range_stop=5):
    for image in original_files:
        im = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        for i in np.arange(start, stop, step):
            img_rs = cv2.resize(im, (0, 0), fx=i, fy=i)

            for j in range(1, range_stop, 1):
                cv2.imwrite(join(save, basename(image)[:-4] + f"-resize_{round(i, 1)}_{j}.png"), img_rs)


def cutout(im, is_pil=False):
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
