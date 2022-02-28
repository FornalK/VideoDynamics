import argparse
import os
import sys
import subprocess
from zipfile import ZipFile
import sys
import csv
import numpy as np
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
from os.path import isfile, join
import datetime as dt
from datetime import date
from datetime import datetime
from google.colab import drive
from google.colab.patches import cv2_imshow
from tqdm import tqdm
import re
import torch
from keras.preprocessing import image
from keras.models import model_from_json

def get_emotions(input_path, out_path, name, skip_frames=10):
  # read video file
  cap = cv2.VideoCapture(input_path)
  
  # load preconfigured model and precomputed weights
  model = model_from_json(open("/content/downloads/facial_expression_model_structure.json", "r").read())         ## REQUIRED TO CHANGE PATH
  model.load_weights('/content/downloads/facial_expression_model_weights.h5')                                    ## REQUIRED TO CHANGE PATH
  face_cascade = cv2.CascadeClassifier('/content/downloads/haarcascade_frontalface_default.xml')    ## REQUIRED TO CHANGE PATH
  
  df = pd.DataFrame(columns=['TimeStamp','Frame','Number of Faces (left->right)','Face Position','angry','disgust','fear','happy','sad','surprise','neutral'])
  
  try:
      i = 0
      while (cap.isOpened()):
          # cap.read() -> checks whether or not frame has been read correctly. Returns boolean value
          ret, frame = cap.read()
          if not (ret):
              break
  
          # for every x frame
          if cap.get(cv2.CAP_PROP_POS_FRAMES) % skip_frames == 0:
              gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
              faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  
              face_counter = 0
              
              for (x, y, w, h) in faces:
                  # cuts frames into required data (requires 4-Dim instead of 3, therefore i used the given approach from the project)
                  cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                  detected_face = frame[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
                  detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
                  detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48x48
                  img_pixels = image.img_to_array(detected_face)
                  img_pixels = np.expand_dims(img_pixels, axis=0)/255
  
                  # prediction
                  predictions = model.predict(img_pixels)

                  df.loc[i,'TimeStamp'] = cap.get(cv2.CAP_PROP_POS_MSEC)
                  df.loc[i,'Frame'] = cap.get(cv2.CAP_PROP_POS_FRAMES)
                  df.loc[i,'Number of Faces (left->right)'] = len(faces)
                  df.loc[i,'Face Position'] = face_counter
                  df.loc[i,'angry':] = pd.Series(predictions[0],index=df.loc[:,'angry':].columns)

                  face_counter += 1
                  i +=1

      # When everything done, release the capture
      cap.release()
      cv2.destroyAllWindows

      df.to_csv(out_path + name + '_FrameLevel_emotions.csv')
      pd.concat([df.loc[:,'angry':].mean(),df.loc[:,'angry':].max()],axis=1).rename(columns={0:'Average',1:'Max'}).to_csv(out_path + name + '_VideoLevel_emotions.csv')
  
      return True
  except:
      return sys.exc_info()[1]


# loop through video and get brightness and colors
def color_loop(input_path, output_path, name, w2c_folder, skip_frame=5):

    w2c = np.array(open(w2c_folder, "rt").read().splitlines())
    cap = cv2.VideoCapture(input_path)
    col_stats = pd.DataFrame(
        columns=['frame_num', 'colorfulness', 'saturation', 'value', 'black', 'blue', 'brown', 'grey', 'green',
                 'orange', 'pink', 'purple', 'red', 'white', 'yellow'])

    try:
        while (True):
            ret, frame = cap.read()
            if not (ret): break

            # get time and frame count and all color frames
            time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)

            if frame_count % skip_frame == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # get resolution, aspect ratio
                resolution = [frame.shape[0], frame.shape[1]]

                # get values for saturation, value, colorfulness
                saturation, value = basic_cols(hsv_frame)
                colorfulness = image_colorfulness(rgb_frame)

                # get exact colors
                img_vec = np.reshape(rgb_frame, (-1, 3))
                black, blue, brown, grey, green, orange, pink, purple, red, white, yellow = get_color_share(w2c,
                                                                                                            img_vec)
                col_stats = col_stats.append(pd.DataFrame([[frame_count, saturation, value, colorfulness, black, blue,
                                                            brown, grey, green, orange, pink, purple, red, white,
                                                            yellow]],
                                                          columns=['frame_num', 'colorfulness', 'saturation', 'value',
                                                                   'black', 'blue', 'brown', 'grey', 'green', 'orange',
                                                                   'pink', 'purple', 'red', 'white', 'yellow']),
                                             ignore_index=True)

        cap.release()
        cv2.destroyAllWindows()

        # here df aggregation mean

        write_csv(col_stats, output_path, name + '_FrameLevel_colors.csv')
        write_csv(resolution, output_path, name + '_VideoLevel_resolution.csv')
        color_columns = ['colorfulness', 'saturation', 'value', 'black', 'blue', 'brown', 'grey', 'green', 'orange',
                         'pink', 'purple', 'red', 'white', 'yellow']
        
        vl_col_stats = pd.DataFrame(pd.concat([col_stats[color_columns].mean(axis=0),col_stats[color_columns].var(axis=0)],axis=1),index=color_columns).rename(columns={0:'mean',1:'variance'})

        write_csv(vl_col_stats, output_path, name + '_VideoLevel_colors.csv', index = True)
        return True
    except:
        return sys.exc_info()[1]


# get video length
def vid_length(input_path, output_path, vid_id):
    try:
        clip = VideoFileClip(input_path)
        write_csv(clip.duration, output_path, vid_id + '_VideoLevel_length.csv')
        try:
            clip.reader.close()
        except:
            pass
        try:
            clip.audio.reader.close_proc()
        except:
            pass
        return True
    except:
        return sys.exc_info()[1]

# get visual variance
def get_visual_variance(input_path, output_path, name, batch_size=16, num_workers=4):
    try:
        
        # get cuts to folder
        get_cuts(input_path, output_path, name, save_img = True)

        dataset = {'embed' : datasets.ImageFolder('/content/scene_imgs/' + name, data_transforms['embed'])}
        dataloader = {'embed': torch.utils.data.DataLoader(dataset['embed'], batch_size = batch_size, shuffle=False, num_workers=num_workers)}

        outputs = torch.empty(0, 2048).to(device)
        for inputs, _ in dataloader['embed']:
            inputs = inputs.to(device)
            output = resnet152(inputs)
            output = torch.reshape(output,output.shape[:2])
            outputs = torch.cat((outputs,output), 0)

        nvec = norml2(outputs)
        D = nvec.mm(torch.t(nvec))
        D = D.to('cpu').numpy()
        neighbor_sim = np.diag(D,k=1)
        n = D.shape[0]
        D_mean = (D.mean()*n-1)/(n-1)
        D_var = np.sqrt((np.square(D).sum()-2*D_mean*(D.sum())+D_mean**2*n**2-n*(1-D_mean)**2)/(n*(n-1)))

        np.savetxt(output_path + name + '_FrameLevel_embeddings.csv',outputs.to('cpu').numpy(),delimiter=',')
        np.savetxt(output_path + name + '_FrameLevel_similarities.csv',D,delimiter=',')
        np.savetxt(output_path + name + '_FrameLevel_similarities_neighbor.csv', neighbor_sim, delimiter=",")

        with open(output_path + name + '_VideoLevel_similarities_all.csv','w') as f:
          f.write('Variable,Value\n')
          f.write('Average_Scene2Scene_Similarity,' +str(neighbor_sim.mean()) + '\n')
          f.write('Variance_Scene2Scene_Similarity,' +str(neighbor_sim.var()) + '\n')
          f.write('Average_Scenes_Similarity,' +str(D_mean) + '\n')
          f.write('Variance_Scenes_Similarity,' +str(D_var))
        return True
        
    except:
        return sys.exc_info()[1]

# get scene cuts
def get_cuts(input_path, output_path, name, save_img=False, del_row=True):
    col_stats = pd.DataFrame(
        columns=['frame_num', 'colorfulness', 'saturation', 'value', 'black', 'blue', 'brown', 'grey', 'green',
                 'orange', 'pink', 'purple', 'red', 'white', 'yellow'])
    try:

        # get command
        command = 'scenedetect --input ' + '"' + input_path + '"' + ' --output ' + '"' + output_path + '"' + ' detect-content list-scenes -f ' + name + '_FrameLevel_scenes.csv -q '
        print(command)
        if save_img: 
            out_folder = '/content/scene_imgs/' + name + '/frames/'
            if not os.path.exists(out_folder): os.makedirs(out_folder)
            command = command + 'save-images -o ' + out_folder

        # run command
        subprocess.call(command, shell=True)

        # delete top row as its filled with unnecessary time code for video split
        if del_row: del_top_row_csv(str(output_path + name + '_FrameLevel_scenes.csv'))

        # calculate aggregated feature avg_scene_freq
        data = pd.read_csv(output_path + name + '_FrameLevel_scenes.csv')
        avg_scene_freq = data['Scene Number'].max() / data['End Time (seconds)'].max()
        write_csv(avg_scene_freq, output_path, name + '_VideoLevel_scenes_freq.csv')

        #only keep middle picture
        if save_img:
          for i in os.listdir(out_folder):
            if i.split('.jpg')[0][-2:] != '02':
              os.remove(out_folder + i)

        return True

    except:
        return sys.exc_info()[1]

#get quality as variance of laplacian of grayscaled image
def get_quality(input_path, output_path, name, skip_frames):
    cap = cv2.VideoCapture(input_path)
    try:
        qual_stats = pd.DataFrame(columns=['frame_num', 'quality'])
        while (cap.isOpened()):
            # cap.read() -> checks whether or not frame has been read correctly. Returns boolean value
            ret, frame = cap.read()
            if not (ret):
                break

            # for every x frame
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % skip_frames == 0:
                #convert image to grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)

                qual_stats = qual_stats.append(pd.DataFrame([[frame_count, cv2.Laplacian(frame, cv2.CV_64F).var()]],
                                                              columns=['frame_num', 'quality']),ignore_index=True, sort=False)

        cap.release()
        cv2.destroyAllWindows()

        # here df aggregation mean

        write_csv(qual_stats, output_path, name + '_FrameLevel_quality.csv')
        write_csv(qual_stats.quality.mean(), output_path, name + '_VideoLevel_quality.csv')
        
        return True
    except:
        return sys.exc_info()[1]

# deletes the top rows until row index = x in file
def del_top_row_csv(file, x=0):
    with open(file, "r") as f:
        lines = csv.reader(f, delimiter=",")  # , is default
        rows = list(lines)
        del (rows[x])

    with open(file, 'w', newline='\n') as f2:
        cw = csv.writer(f2, delimiter=',')
        cw.writerows(rows)


# write csv file independent of type of datainput (df, str)
def write_csv(content, folder, name, index=False, header=True, mode='w'):
    try:
        content.to_csv(folder + name, index=index, header=header, mode=mode)
    except:
        try:
            with open(str(folder) + str(name), mode=mode) as output:
                output.write(str(content))
        except:
            print('csv creation failed for: ', folder, name)

        # create a folder for each video to store extracted feature information


def create_folder(location):
    try:
        os.mkdir(location)
    except:
        pass


# get face information by looping through faces
def video_loop_faces(input_path, output_path, name, skip_frame=10):
    from mtcnn.mtcnn import MTCNN
    detector = MTCNN()

    cap = cv2.VideoCapture(input_path)

    face_file = pd.DataFrame(columns=['time', 'frame', 'num_faces', 'faces'])

    write_csv(face_file, output_path, name + '_FrameLevel_faces.csv')

    try:
        while (True):
            ret, frame = cap.read()
            if not (ret): break

            if cap.get(cv2.CAP_PROP_POS_FRAMES) % skip_frame == 0:
                # get face vector from frame
                contain = detector.detect_faces(frame)
                count = len(contain)

                row = str(round(cap.get(cv2.CAP_PROP_POS_MSEC), 0)) + ',' + str(
                    cap.get(cv2.CAP_PROP_POS_FRAMES)) + ',' + str(count) + ',"' + str(contain) + '"\n'
                write_csv(row, output_path, name + '_FrameLevel_faces.csv', mode='a')

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

        # get aggregated variables avg_face_num and face_ratio
        df = pd.read_csv(output_path + name + '_FrameLevel_faces.csv')
        face_ratio = df[df['num_faces'] != 0].shape[0] / df.shape[0]
        avg_num_faces = df[df['num_faces'] != 0]['num_faces'].mean()

        write_csv(face_ratio, output_path, name + '_VideoLevel_faces_ratio.csv')
        write_csv(avg_num_faces, output_path, name + '_VideoLevel_faces_avg_number.csv')

        return True
    except:
        return sys.exc_info()[1]


# format frames for input to coco
def LoadImages2(img0,img_size):
    # Padded resize
    img = letterbox(img0, new_shape=img_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)  # uint8 to fp16/fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    return img


# get 80 coco objet information
def coco_loop(input_path, output_path, name, yolo_img_size,conf_thres,iou_thres, skip_frame=10):
    cap = cv2.VideoCapture(input_path)
    
    try:
      with open(output_path + name + '_FrameLevel_objects.csv', 'w') as file:
          file.write('frame,y1,x1,y2,x2,object,confidence\n')

          while (True):
              ret, frame = cap.read()
              if not (ret): break

              if cap.get(cv2.CAP_PROP_POS_FRAMES) % skip_frame == 0:
                  img = LoadImages2(frame,yolo_img_size)
                  img = torch.from_numpy(img).to(device)

                  # what does this do?
                  if img.ndimension() == 3:
                      img = img.unsqueeze(0)
                  pred = coco_model(img)[0]
                  # Apply NMS
                  pred = non_max_suppression(pred, conf_thres, iou_thres, multi_label=False)
                  # Process detections
                  for i, det in enumerate(pred):  # detections per image
                      if det is not None and len(det):
                          # Rescale boxes from img_size to im0 size
                          det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                          # Write results

                          for *xyxy, conf, cls in det:
                              file.write(('%d,' + '%g,' * 4 + '%s,%g\n') % (cap.get(cv2.CAP_PROP_POS_FRAMES), *xyxy, names[int(cls)], conf))

          frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
          width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
          height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
          # When everything done, release the capture
          cap.release()
          cv2.destroyAllWindows()
      df = pd.read_csv(output_path + name + '_FrameLevel_objects.csv')

      df['area'] = (df['x2'] - df['x1']) * (df['y2'] - df['y1']) / width / height
      with open(output_path + name + '_VideoLevel_objects_human_area.csv', 'w') as file:
          file.write('Human Area Coverage\n')
          file.write(str(df[df['object'] == 'person']['area'].sum() / int(frames / skip_frame)))
      return True

    except:
        return sys.exc_info()[1]


# get colorfulness from image in RGB, frame needs to be passed
def image_colorfulness(image):
    # using Hasler, David ; Süsstrunk, Sabine, 2003: Measuring colourfulness in natural images

    # split the image into its respective RGB components
    R = image[:, 0]
    G = image[:, 1]
    B = image[:, 2]

    # compute rg = R - G
    rg = R - G

    # compute yb = 0.5 * (R + G) - B
    yb = 0.5 * (R + G) - B

    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    # derive the "colorfulness" metric and return it
    return (stdRoot + (0.3 * meanRoot)) / 255

# get an image in HSV space , frame needs to be passed
def HSVmetrics(hsvim):
    # hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    width, height, channels = hsvim.shape
    # print(hsvim.shape)

    # get Hh
    H = hsvim[:, :, 0]
    A = np.sum(np.cos(np.pi * 2 * H / 180))
    B = np.sum(np.sin(np.pi * 2 * H / 180))
    Hh = np.arctan(B / A) / np.pi
    if A > 0 and B < 0:
        Hh += 2
    elif A < 0:
        Hh += 1

    # get V
    V = 1 - (np.sqrt(A ** 2 + B ** 2) / height / width)

    # get Hs
    S = hsvim[:, :, 1]
    As = np.sum(S * np.cos(np.pi * 2 * H / 180)) / 255
    Bs = np.sum(S * np.sin(np.pi * 2 * H / 180)) / 255
    if As == 0:
        Hs = 0
    else:
        Hs = np.arctan(Bs / As) / np.pi
    if As > 0 and Bs < 0:
        Hs += 2
    elif As < 0:
        Hs += 1

    # get meanS, stdS, warmth, saturation_scaled_warmth
    meanS = np.mean(S)
    stdS = np.std(S)
    warmth = A / width / height
    saturation_scaled_warmth = As / width / height
    # Rn= np.sqrt(As**2+Bs**2)/len(S)**2

    # get Rs
    sumS = np.sum(S)
    if sumS == 0:
        Rs = 0
    else:
        Rs = np.sqrt(As ** 2 + Bs ** 2) / np.sum(S) * 255

    # get meanV, stdV
    Value = hsvim[:, :, 2]
    meanV = np.mean(Value) / 255
    stdV = np.std(Value) / 255

    # get pleasure, arousal, dominance
    pleasure = np.mean(0.69 * Value[:] + 0.22 * S[:])
    arousal = np.mean(0.31 * Value[:] + 0.60 * S[:])
    dominance = np.mean(-0.76 * Value[:] + 0.32 * S[:])

    return Hh / 2, V, warmth, saturation_scaled_warmth, meanS / 255, stdS / 255, Hs, Rs, meanV, stdV, pleasure, arousal, dominance


# identify colors defined by additional w2c file, frame needs to be passed
def image2colors(w2c, img_vec):
    img_vec = np.reshape(img_vec, (-1, 3))
    colors = w2c[np.array(
        np.floor(img_vec[:, 0] / 8) + 32 * np.floor(img_vec[:, 1] / 8) + 1024 * np.floor(img_vec[:, 2] / 8)).astype(
        int)]
    unique, counts = np.unique(colors, return_counts=True)
    counts = counts / len(img_vec)
    return unique, counts


# get colors dataframe defined by additional w2c file, frame needs to be passed
def get_color_share(w2c, img_vec):
    unique, counts = image2colors(w2c, img_vec)

    color = []
    for i in range(1, 12):
        if counts[np.where(unique == str(i))[0]]:
            color.append(round(np.array(counts[np.where(unique == str(i))])[0], 3))
        else:
            color.append(0)
    return color


# get basic color stats incl RGB, brightness and colorfulness, frame needs to be passed
def basic_cols(hsv_frame):
    saturation = np.rint(np.average(hsv_frame[:, :, 1])) / 255
    value = np.rint(np.average(hsv_frame[:, :, 2])) / 255

    return saturation, value

def install_dependencies_cuts():
    with ZipFile('/content/downloads/PySceneDetect-master.zip', 'r') as zipObj:
        zipObj.extractall('/content/downloads/')
        os.chdir('/content/downloads/PySceneDetect-master/')
        subprocess.getoutput('python setup.py install')
        os.chdir('/content/')
        
def install_dependencies_yolov3():
    with ZipFile('/content/downloads/yolov3-master.zip', 'r') as zipObj:
        zipObj.extractall('/content/downloads/')
        os.chdir('/content/downloads/yolov3-master/')
        subprocess.getoutput('pip install -U -r requirements.txt')
        os.chdir('/content/')


def install_dependencies_faces():
  subprocess.getoutput('pip install mtcnn')

def install_dependencies_variance():
  install_dependencies_cuts()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__' and torch.cuda.is_available():
    parser = argparse.ArgumentParser()

    parser.add_argument('--extract-length', type=str2bool, default=False) # set confidence threshold for coco object detecion
    parser.add_argument('--extract-cuts', type=str2bool, default=False) # set confidence threshold for coco object detecion
    parser.add_argument('--extract-colors', type=str2bool, default=False) # set confidence threshold for coco object detecion
    parser.add_argument('--extract-faces', type=str2bool, default=False) # set confidence threshold for coco object detecion
    parser.add_argument('--extract-emotions', type=str2bool, default=False) # set confidence threshold for coco object detecion
    parser.add_argument('--extract-objects', type=str2bool, default=False) # set confidence threshold for coco object detecion
    parser.add_argument('--extract-variance', type=str2bool, default=False) # set confidence threshold for coco object detecion
    parser.add_argument('--extract-quality', type=str2bool, default=False) # set confidence threshold for coco object detecion
    parser.add_argument('--start-index', type=int, default=0) 
    parser.add_argument('--end-index', type=int, default='last') # set confidence threshold for coco object detecion
    parser.add_argument('--in-folder', type=str, default='/content/drive/My Drive/trailer/vids2/')  # set overlay threshold for coco object detecion
    parser.add_argument('--out-folder', type=str, default='/content/drive/My Drive/trailer/preds/')  # set overlay threshold for coco object detecion
    parser.add_argument('--log-name', type=str, default=date.today().strftime("%Y-%m-%d")+'_logfile.csv', help = 'name of lofile, pls include .csv ending')  # set overlay threshold for coco object detecion
    parser.add_argument('--log-folder', type=str, default='/content/drive/My Drive/trailer/logs/', help='folder for logfile')  # set overlay threshold for coco object detecion
    parser.add_argument('--scene-imgs-path', type=str, default='/content/drive/My Drive/trailer/preds/scenes/')  # set overlay threshold for coco object detecion
    parser.add_argument('--skip_frame', type=int, default=10, help = 'Show only every n-th frame; enter n')  
    parser.add_argument('--yolo-img-size', type=str, default=(608, 352)) # image size for coco detection - ATTENTION if changed, different pretrained file required!
    parser.add_argument('--yolo-cfg', type=str, default='/content/downloads/yolov3-master/cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--yolo-names', type=str, default='/content/downloads/yolov3-master/data/coco.names', help='*.names path')
    parser.add_argument('--yolo-folder', type=str, default='/content/downloads/yolov3-master/', help='yolo-master path')
    parser.add_argument('--yolo-weight', type=str, default='/content/downloads/yolov3-spp-ultralytics.pt', help='*.pt path')
    parser.add_argument('--yolo-conf-thres', type=float, default=0.3) # set confidence threshold for coco object detecion
    parser.add_argument('--yolo_iou_thres', type=float, default=0.5) # set confidence threshold for coco object detecion
    parser.add_argument('--visual-variance-num-workers', type=int, default=4, help='*.pt path')
    parser.add_argument('--visual-variance-batch-size', type=int, default=16, help='*.pt path')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--w2c_folder', default='/content/downloads/rgb2colors.csv', help='vector file for color determination')
    parser.add_argument('--images2text-text-folder', default='/content/image2text/text/', help='folder where to save extracted text')
    parser.add_argument('--images2text-images-folder', default='/content/image2text/images/', help='folder where to save and retrieve frames with text')
    opt = parser.parse_args()

    # get list of videos
    vid_files = os.listdir(opt.in_folder)
    vid_files = [file for file in vid_files if file.split('.')[-1] in ['flv', 'f4v', 'f4p', 'f4a', 'f4b', 'nsv', 'roq', 'mxf', '3g2', '3gp', 'svi', 'm4v', 'mpg', 'mpeg', 'm2v', 'mpg', 'mp2', 'mpeg', 'mpe', 'mpv', 'mp4', 'm4p', 'm4v', 'amv', 'asf', 'rmvb', 'rm', 'yuv', 'wmv', 'mov', 'qt', 'MTS', 'M2TS', 'TS', 'avi', 'mng', 'gifv', 'gif', 'drc', 'ogv', 'ogg', 'vob', 'flv', 'flv', 'mkv']]

    if opt.end_index == 'last' or opt.end_index > len(vid_files):
        opt.end_index = len(vid_files)
    if (type(opt.start_index)!=int) or opt.start_index<0 or opt.start_index>opt.end_index:
        print('Setting start-index to 0')
        opt.start_index = 0
    if opt.extract_variance and opt.extract_cuts:
        print('Visual variance requires scene cuts and will automatically extract scene cuts, setting scene_cuts to False')
        opt.extract_cuts = False

    # create output folder and logfiles to save progress and results and track errors
    if not os.path.exists(opt.out_folder): os.makedirs(opt.out_folder)
    if not os.path.exists(opt.log_folder): os.makedirs(opt.log_folder)
    logcols = ['time','name','log.length','log.cuts','log.colors','log.faces','log.emotions','log.objects','log.variance','log.quality']
    pd.DataFrame(columns=logcols).to_csv(opt.log_folder+opt.log_name,index_label=False)

    from IPython.utils import io
    with io.capture_output() as captured:
         
        if opt.extract_cuts: install_dependencies_cuts()
        if opt.extract_faces: install_dependencies_faces()
        if opt.extract_variance: install_dependencies_variance()

        # load yolov3 model for coco object detection
        if opt.extract_objects:
        
            # unzip yolov3-master.zip
            install_dependencies_yolov3()
            
            #load yolov3 modules
            from sys import platform
            import torch
            sys.path.insert(0, opt.yolo_folder)
            from models import * 
            from utils.datasets import *
            from utils.utils import *
            device = torch_utils.select_device(opt.device)          
            torch.no_grad()
            coco_model = Darknet(opt.yolo_cfg, opt.yolo_img_size)
            coco_model.load_state_dict(torch.load(opt.yolo_weight, map_location=device)['model'])
            coco_model.to(device).eval()
            # Get coco names
            names = load_classes(opt.yolo_names)
            # use half precision for yolov3
            half=False
            if half:
                coco_model.half()
                
        # create visual variance model
        if opt.extract_variance:
            import torch
            import torch.nn as nn
            import torchvision.models as torch_models
            from torch.autograd import Variable
            from torchvision import transforms, datasets
            device = torch.device("cuda:0")
            resnet152 = torch_models.resnet152(pretrained=True)
            resnet_modules = list(resnet152.children())[:-1] # get list of all but last layer from resnet152
            resnet152 = nn.Sequential(*resnet_modules) #recompose list of layers to neural net
            resnet152.to(device)
            for p in resnet152.parameters(): #set all layers in evaluation mode (no derivatives required)
                p.requires_grad = False
            data_transforms = {
                'embed': transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                }

            def norml2(vec):# input N by F
                F = vec.size(1)
                w = torch.sqrt((torch.t(vec.pow(2).sum(1).repeat(F,1))))
                return vec.div(w)

    # start to llop through videos
    print(f'\nAnalysis of {opt.end_index-opt.start_index} videos begins:\n')

    for index in tqdm(range(opt.start_index,opt.end_index)): # Loop through folder with videos to extract selected features
        i = vid_files[index]
        print(f'\n\nAnalyze {i}')
        ### Provide name, input- and output_path
        name = i.split('.')[0]
        input_path = opt.in_folder + i
        output_path = opt.out_folder + name + '/'
        if not os.path.exists(output_path): os.makedirs(output_path)

        with io.capture_output() as captured:

          ### Extract selected features
          if opt.extract_length: log_length = vid_length(input_path,output_path,name)
          else: log_length=False
          if opt.extract_cuts: log_cuts = get_cuts(input_path,output_path,name)
          else:log_cuts=False
          if opt.extract_colors: log_colors = color_loop(input_path,output_path,name,opt.w2c_folder)
          else:log_colors=False
          if opt.extract_faces: log_faces = video_loop_faces(input_path,output_path,name)
          else: log_faces=False
          if opt.extract_emotions: log_emotions = get_emotions(input_path,output_path,name)
          else: log_emotions=False
          if opt.extract_objects: log_objects = coco_loop(input_path,output_path,name,opt.yolo_img_size,opt.yolo_conf_thres,opt.yolo_iou_thres,opt.skip_frame)
          else: log_objects=False
          if opt.extract_variance: log_variance = get_visual_variance(input_path,output_path,name,opt.visual_variance_num_workers,opt.visual_variance_batch_size)
          else: log_variance=False
          if opt.extract_quality: log_quality = get_quality(input_path,output_path,name,opt.skip_frame)
          else: log_quality=False
            
        ### Write potential errors into logfile
        row = pd.Series([datetime.now(),name,log_length,log_cuts,log_colors,log_faces,log_emotions,log_objects,log_variance,log_quality])
        pd.DataFrame([row]).to_csv(opt.log_folder+opt.log_name,mode='a+',header=False,index=None)

else:
    print('please activate GPU: Runtime -> Change runtime type -> GPU')