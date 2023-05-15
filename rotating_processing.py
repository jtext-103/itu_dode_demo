# @Time : 2023/5/12 18:34 
# @Author : zhongyu 
# @File : rotating_processing.py
from jddb.file_repo import FileRepo
from jddb.processor import ShotSet
from util.rotating_mode_std_processor import RotatingModeStd
from sklearn.model_selection import train_test_split
from scipy import interpolate
import matplotlib.pyplot as plt
from jddb.processor.basic_processors import ResamplingProcessor, NormalizationProcessor, ClipProcessor, TrimProcessor
from util.ip_error_process import IpError
import warnings

Mir = [
    "MA_TOR1_R01", "MA_TOR1_R02",
    "MA_POL_CA01T", "MA_POL_CA03T", "MA_POL_CA05T", "MA_POL_CA07T", "MA_POL_CA09T", "MA_POL_CA11T",
    "MA_POL_CA13T", "MA_POL_CA15T", "MA_POL_CA17T", "MA_POL_CA19T", "MA_POL_CA21T", "MA_POL_CA23T"
]

if __name__ == '__main__':
    upload_file_repo = FileRepo('..//FileRepo//upload_file//$shot_2$00//')
    # create a shot set with a file
    source_shotset = ShotSet(upload_file_repo)
    shot_list = source_shotset.shot_list
    processed_shotset = source_shotset.process(
        processor=ResamplingProcessor(50000),
        input_tags=['bt'],
        output_tags=['bt_high'],
        save_repo=upload_file_repo)
    processed_shotset = processed_shotset.process(TrimProcessor(),
                                                  input_tags=[Mir[2:] + ['bt_high']],
                                                  output_tags=[Mir[2:] + ['bt_high']],
                                                  save_repo=upload_file_repo)
    processed_shotset = processed_shotset.process(
        processor=RotatingModeStd(),
        input_tags=[Mir[2:] + ['bt_high']],
        output_tags=['rotating_mode_proxy'],
        save_repo=upload_file_repo)
    processed_shotset = processed_shotset.remove_signal(tags=['bt_high'],
                                                        save_repo=upload_file_repo)
