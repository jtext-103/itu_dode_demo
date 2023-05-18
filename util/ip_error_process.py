# @Time : 2023/5/10 21:54 
# @Author : zhongyu 
# @File : ip_error_process.py
from jddb.processor import Signal, BaseProcessor
import numpy as np
from copy import deepcopy


class IpError(BaseProcessor):
    """
            Given the ip target signal and ip
            signal to calculate the ip error.

    """

    def __init__(self, ):
        super().__init__()

    def transform(self, ip_target_signal: Signal, ip_signal: Signal) -> Signal:
        """

        :param ip_target_signal:
        :param ip_signal:
        :return:Signal, ip error
        """
        resampled_attributes = deepcopy(ip_signal.attributes)
        new_data = (ip_signal.data - ip_target_signal.data) / ip_signal.data

        return Signal(data=new_data, attributes=resampled_attributes)
