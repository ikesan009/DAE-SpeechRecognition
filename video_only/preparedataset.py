#coding:utf-8
import cv2
import os
import numpy as np
import glob
import threading
from tqdm import tqdm_notebook as tqdm

class PrepareDataset(object):
    
    def __init__(self, dir_dataset, dir_save):
        self.dir_dataset = dir_dataset
        self.dir_save = dir_save
    
    #dirで指定されたパスが存在しない場合ディレクトリ作成
    def make_dir(self, dir, format=False):
        if not os.path.exists(dir):
            os.makedirs(dir)
        if format and os.path.exists(dir):
            shutil.rmtree(dir)
            
    #フレーム画像のキャプチャ
    def capture(self, dir_dataset, path_video, dir_save, format_dir=False):
        #print('capturing from video...')
        self.make_dir(dir_save, format_dir)
        cap = cv2.VideoCapture(os.path.join(dir_dataset, path_video))
        i = 0
        array_frame = np.ndarray(shape=(29, 95, 95, 3), dtype='uint8')
        while(cap.isOpened()):
            flag, frame = cap.read()
            if (flag==False):
                break
            frame = frame[116:211,80:175]
            array_frame[i] = frame
            """
                basename = os.path.basename(path_video)
                fname = os.path.splitext(basename)[0] + str(i).zfill(4) + '.png'
                path_save = os.path.join(dir_save, fname)
                cv2.imwrite(path_save, frame)
            """
            i += 1
            basename = os.path.basename(path_video)
            fname = os.path.splitext(basename)[0] + '.npy'
            path_save = os.path.join(dir_save, fname)
            np.save(path_save, array_frame)
        #print('Done.')
        cap.release()

    #スレッド処理
    def th_capture(self, dir_dataset, pathlist_video, dir_save, name):
        pathlist_video = tqdm(pathlist_video)
        pathlist_video.set_description("thread :"+str(name))
        for path_video in pathlist_video:
            dir_save_ = os.path.join(dir_save, os.path.dirname(path_video))
            self.capture(dir_dataset, path_video, dir_save_, format_dir=False)

    #キャプチャの実行
    def run_capture(self, num_thread):
        print('running capture...')

        os.chdir(self.dir_dataset)
        pathlist_video = glob.glob('./*'+'/*'+'/*.mp4')
        os.chdir('../')
        step_th = int(len(pathlist_video) / num_thread)
        rem = len(pathlist_video) % num_thread
        threadlist = []

        for i in range(0, num_thread-1):
            thread = threading.Thread(
                target=self.th_capture,
                args=([self.dir_dataset, pathlist_video[i*step_th:(i+1)*step_th], self.dir_save, i])
            )
            threadlist.append(thread)
        thread = threading.Thread(
            target=self.th_capture,
            args=([self.dir_dataset, pathlist_video[(num_thread-1)*step_th:num_thread*step_th+rem], self.dir_save, num_thread-1])
        )
        threadlist.append(thread)
        for thread in threadlist:
            thread.start()
            
        print('capture was done.')