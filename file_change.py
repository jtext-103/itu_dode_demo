# @Time : 2023/4/3 14:08 
# @Author : zhongyu 
# @File : file_change.py
import numpy as np
import xlrd

from jddb.file_repo import FileRepo

if __name__ == '__main__':
    # shot list & tag list
    excel_path = r"..\meta\disruptions1.xlsx"  #
    dp_book = xlrd.open_workbook(excel_path, encoding_override="utf-8")
    dp_sheet = dp_book.sheet_by_index(0)

    tags = list(hdf5ReaderConfig2A.dir.keys())
    nrowsum = dp_sheet.nrows
    change_file_repo = FileRepo("..//file_repo//ChangeShots//$shot_2$XX//$shot_1$X//")
    for i in range(0, nrowsum):
        meta_result = dict()
        data_result = dict()
        shot = int(dp_sheet.row_values(i)[0])
        change_file_repo.create_shot(shot)
        for channel in tags:
            cha_out = h5r.if_channel_exist(shot, channel, '2A')
            if not cha_out:
                meta_result[channel] = 0
            else:
                meta_result[channel] = 1
                time, data_shot = h5r.read_channel(shot, channel, device="2a")
                freq = h5r.get_attrs("SampleRate", shot_number=shot, channel=channel)
                start_t = h5r.get_attrs("StartTime", shot_number=shot, channel=channel)
                data_dict = {channel: data_shot}
                if 'EFIT' in channel:
                    attrs_dict = {"SampleRate": freq, "StartTime": start_t}
                else:
                    attrs_dict = {"SampleRate": freq * 1000, "StartTime": start_t}
                if channel == 'IP_TARGET':
                    attrs_dict = {"SampleRate": freq * 1000, "StartTime": start_t / 1000}

                # data & attributes write
                change_file_repo.write_data_file(change_file_repo.get_file(shot), data_dict)
                change_file_repo.write_attributes(shot, channel, attrs_dict)
        # meta
        meta_result['DownTime'] = dp_sheet.row_values(i)[2] / 1000
        if dp_sheet.row_values(i)[1]:
            meta_result['IsDisrupt'] = 1
        else:
            meta_result['IsDisrupt'] = 0
        change_file_repo.write_label(shot, meta_result)
