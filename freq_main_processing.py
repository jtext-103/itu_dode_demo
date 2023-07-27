# @Time : 2023/5/11 22:29 
# @Author : zhongyu 
# @File : freq_main_processing.py
import numpy as np
from jddb.file_repo import FileRepo
from jddb.processor import ShotSet
from util.basic_processor import SliceProcessor, FFTProcessor, find_tags, AlarmTag, get_machine_tags
from jddb.processor.basic_processors import ResamplingProcessor,TrimProcessor
import pandas as pd

if __name__ == '__main__':
    # %%
    # choose common signal
    df_signal = pd.read_csv('ITU data - signal.csv')  # read signal csv file
    common_signal = []
    # check every signal, if it isn't nan in 3 machine signal list, it should be included in common_signal.
    for signal_name in df_signal.Diagnostics:
        target_row = df_signal.loc[df_signal.Diagnostics == signal_name]
        if ~target_row['J-TEXT MDSplus Tag'].isna().values[0] and \
                ~target_row['C-Mod MDSplus Tag'].isna().values[0]:
            common_signal.append(signal_name)
    # change common_signal to different sub list according to whether it belongs to array mirnove, sxr, axuv.
    mir_name_list = find_tags('poloidal', common_signal) + find_tags('toroidal Mir', common_signal)
    sxr_name_list = find_tags('soft', common_signal)
    axuv_name_list = find_tags('AXUV', common_signal)
    basic_name_list = list(set(common_signal) - set(mir_name_list + sxr_name_list + axuv_name_list))
    # choose machine for processing and get machine tags by common name list
    machine_name = 'J-TEXT'
    basic_machine_tags = get_machine_tags(machine_name, basic_name_list, df_signal)
    mir_machine_tags = get_machine_tags(machine_name, mir_name_list, df_signal)
    # %%
    source_file_repo = FileRepo('..//FileRepo//processed_zy//$shot_2$00//')
    train_file_repo = FileRepo('..//FileRepo//train_file//$shot_2$00//')
    # create a valid shot set with a file, the valid shots should contain target tags and enough flattop time.
    source_shotset = ShotSet(source_file_repo)
    shot_list = source_shotset.shot_list
    # Define the target tags which contain signals for process
    targ_tags = basic_machine_tags + mir_machine_tags
    valid_shots = []  # Initialize an empty list to store valid shots
    for shot in shot_list:
        all_tags = list(source_shotset.get_shot(shot).tags)
        last_time = list(source_file_repo.read_labels(shot, ['DownTime']).values())
        # Check if all target tags are present in the shot's tags and last_time of shot is greater than 0.2s
        if all(tag in all_tags for tag in targ_tags) & (last_time[0] > 0.2):
            valid_shots.append(shot)
    valid_shotset = ShotSet(source_file_repo, valid_shots)  # Create a new ShotSet object using the valid shots

    # 1. FFT processing for max frequency and amplitude of mirnov signals
    # %%
    for signal_name in mir_name_list:
        # slicing
        # get mir signal tags of machine
        target_row = df_signal.loc[df_signal.Diagnostics == signal_name]
        mir_tag = target_row['{} MDSplus Tag'.format(machine_name)].values[0]
        processed_shotset = valid_shotset.process(
            processor=SliceProcessor(window_length=250, overlap=0.9),
            input_tags=[mir_tag],
            output_tags=["sliced_MA_{}".format(mir_tag)],
            save_repo=train_file_repo)
        # fft MA
        processed_shotset = processed_shotset.process(
            processor=FFTProcessor(),
            input_tags=["sliced_MA_{}".format(mir_tag)],
            output_tags=[["fft_amp_{}".format(mir_tag), "fft_fre_{}".format(mir_tag)]],
            save_repo=train_file_repo)

    # %%
    # 2. remove redundant tags and keep tags for model training
    shot_list = processed_shotset.shot_list
    all_tags = list(processed_shotset.get_shot(shot_list[0]).tags)
    fft_tag = find_tags('fft_', all_tags)  # tags coming from process should be included like fft and so on.
    keep_tags = basic_machine_tags + fft_tag  # mir signals have been processed, they shouldn't be kept.
    processed_shotset = processed_shotset.remove_signal(tags=keep_tags, keep=True,
                                                        save_repo=train_file_repo)

    # %%
    # 3. resample high frequency tags
    down_tags = fft_tag
    processed_shotset = processed_shotset.process(
        processor=ResamplingProcessor(1000),
        input_tags=down_tags,
        output_tags=down_tags,
        save_repo=train_file_repo)

    # %%
    # 4. trim  signal
    common_tags = basic_name_list + fft_tag  # change the machine tags to common names
    processed_shotset = processed_shotset.process(
        TrimProcessor(),
        input_tags=[keep_tags],
        output_tags=[common_tags],
        save_repo=train_file_repo)
    keep_tags = basic_name_list + fft_tag  # keep common names
    processed_shotset = processed_shotset.remove_signal(tags=keep_tags, keep=True,
                                                        save_repo=train_file_repo)

    # %%
    # 5. add disruption labels for each time point as a signal called alarm_tag
    processed_shotset = processed_shotset.process(
        processor=AlarmTag(lead_time=0.1, disruption_label="IsDisrupt", downtime_label="DownTime"),
        input_tags=["plasma current"],
        output_tags=["alarm_tag"],
        save_repo=train_file_repo)
