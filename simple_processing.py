# @Time : 2023/5/11 21:48 
# @Author : zhongyu 
# @File : simple_processing.py
from MDSplus import connection
import numpy as np
from jddb.file_repo import FileRepo
from jddb.processor import ShotSet
from sklearn.model_selection import train_test_split
from scipy import interpolate
import matplotlib.pyplot as plt
from jddb.processor.basic_processors import ResamplingProcessor, NormalizationProcessor, ClipProcessor, TrimProcessor
from util.ip_error_process import IpError
import warnings

if __name__ == '__main__':
    source_file_repo = FileRepo('..//FileRepo//processed_zy//$shot_2$00//')
    upload_file_repo = FileRepo('..//FileRepo//upload_file//$shot_2$00//')
    # create a shot set with a file
    source_shotset = ShotSet(source_file_repo)
    shot_list = source_shotset.shot_list
    is_disrupt = []
    for threshold in shot_list:
        dis_label = source_file_repo.read_labels(threshold, ['IsDisrupt'])
        is_disrupt.append(dis_label['IsDisrupt'])

    train_shots, test_shots, _, _ = \
        train_test_split(shot_list, is_disrupt, test_size=0.5,
                         random_state=1, shuffle=True, stratify=is_disrupt)
    valid_shotset = ShotSet(source_file_repo, test_shots)
    processed_shotset = valid_shotset.remove_signal(tags=['ip_target'],
                                                    save_repo=upload_file_repo)
