# @Time : 2023/5/10 20:38 
# @Author : zhongyu 
# @File : ip_set_construct.py
from MDSplus import connection
import numpy as np
from jddb.file_repo import FileRepo
from jddb.processor import ShotSet
from scipy import interpolate
import matplotlib.pyplot as plt
from jddb.processor.basic_processors import ResamplingProcessor, NormalizationProcessor, ClipProcessor, TrimProcessor
from util.ip_error_process import IpError
import warnings

c = connection.Connection('222.20.94.136')


def fetch_data(tag):
    tag_data = np.array(c.get(tag))  # diagnostic data
    tag_time = np.array(c.get(r'DIM_OF(BUILD_PATH({}))'.format(tag)))  # DIM_OF(tag), time axis
    return tag_data, tag_time


if __name__ == '__main__':
    scs_process_file_repo = FileRepo("..//FileRepo//processed_scs//$shot_2$00//")
    zy_process_file_repo = FileRepo("..//FileRepo//processed_zy//$shot_2$00//")
    source_shotset = ShotSet(scs_process_file_repo)
    shot_list = source_shotset.shot_list
    for shot in shot_list:
        c.openTree('jtext', shot=shot)
        try:
            ip_targ = np.array(c.get(r'\oh_set'))
            time = np.array(c.get(r'DIM_OF({})'.format(r'\oh_set')))
            f = interpolate.interp1d(time, ip_targ)
            time_new = np.linspace(time[0], time[-1], num=int(1000 * (time[-1] - time[0])))
            ip_targ_new = f(time_new) / 1000
            data_dict = {'ip_target': ip_targ_new}
            attrs_dict = {"SampleRate": 1000, "StartTime": 0}
            scs_process_file_repo.write_data_file(scs_process_file_repo.get_file(shot), data_dict)
            scs_process_file_repo.write_attributes(shot, 'ip_target', attrs_dict)
        except Exception as e:
            warnings.warn("Could not read data from {}".format(shot), category=UserWarning)

    c.closeAllTrees()
    # clip

    processed_shotset = source_shotset.process(processor=ClipProcessor(start_time=0.15, end_time_label="DownTime"),
                                               input_tags=['ip_target'],
                                               output_tags=['ip_target'],
                                               save_repo=zy_process_file_repo)
    # trim part signal
    processed_shotset = processed_shotset.process(TrimProcessor(),
                                                  input_tags=[['ip_target', 'ip']],
                                                  output_tags=[['ip_target', 'ip']],
                                                  save_repo=zy_process_file_repo)
    # ip_error
    processed_shotset = processed_shotset.process(processor=IpError(),
                                                  input_tags=[["ip_target", "ip"]],
                                                  output_tags=['ip_error'], save_repo=zy_process_file_repo)
