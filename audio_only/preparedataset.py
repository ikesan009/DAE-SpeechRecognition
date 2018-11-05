#coding:utf-8
import cv2
import os
import numpy as np
import glob
import subprocess
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
            
    def run_command(self, cmd):
        out = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)       
     
    #mat形式に変換
    def translatetomat(self, dir_dataset, path_audio, dir_save, format_dir=False):
        self.make_dir(dir_save, format_dir)
        basename = os.path.basename(path_audio)
        fname = os.path.splitext(basename)[0] + '.mat'
        path_save = os.path.join(dir_save, fname)
        cmd = "echo \"[audio,Fs] = audioread(\'" \
              + os.path.join(dir_dataset, path_audio) \
              + "\'); save(\'" + path_save \
              + "\');\" | matlab -nodisplay"
        self.run_command(cmd)   
            
    #スレッド処理
    def th_translatetomat(self, dir_dataset, pathlist_audio, dir_save, name):
        pathlist_audio = tqdm(pathlist_audio)
        pathlist_audio.set_description("thread :"+str(name))
        for path_audio in pathlist_audio:
            dir_save_ = os.path.join(dir_save, os.path.dirname(path_audio))
            self.translatetomat(dir_dataset, path_audio, dir_save_, format_dir=False)

    #キャプチャの実行
    def run_translatetomat(self, num_thread):
        print('running translatetomat...')

        os.chdir(self.dir_dataset)
        pathlist_audio = glob.glob('./*'+'/*'+'/*.mp4')
        os.chdir('../')
        step_th = int(len(pathlist_audio) / num_thread)
        rem = len(pathlist_audio) % num_thread
        threadlist = []

        for i in range(0, num_thread-1):
            thread = threading.Thread(
                target=self.th_translatetomat,
                args=([self.dir_dataset, pathlist_audio[i*step_th:(i+1)*step_th], self.dir_save, i])
            )
            threadlist.append(thread)
        thread = threading.Thread(
            target=self.th_translatetomat,
            args=([self.dir_dataset, pathlist_audio[(num_thread-1)*step_th:num_thread*step_th+rem], self.dir_save, num_thread-1])
        )
        threadlist.append(thread)
        for thread in threadlist:
            thread.start()
            
        print('translatetomat was done.')