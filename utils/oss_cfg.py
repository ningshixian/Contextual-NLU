#!/usr/bin/python3  
# -*- coding: utf-8 -*-
# @Time    : 2022/8/11 19:06
# @Author  : wanghao27
# @Site    : 
# @File    : oss_cfg.py
# @Email   : wanghao27@longfor.com
import oss2


class OSS2():
    def __init__(self,config):
        self.tolerant_length=int(config["tolerant_length"])
        _oss_Bucket=config["oss.Bucket"]
        _oss_AccessKeySecret=config["oss.AccessKeySecret"]
        _oss_AccessKey_ID=config["oss.AccessKey_ID"]
        _oss_EndPoint=config["oss.EndPoint"]
        self.oss_large_params_path=config["oss.lp_save_path"]
        self.auth=oss2.Auth(_oss_AccessKey_ID, _oss_AccessKeySecret)
        self.bucket=oss2.Bucket(self.auth, 'http://%s'%_oss_EndPoint,_oss_Bucket)

    
    def oss_put(self,oss_file_path,local_file_path):

        put_oss_resualt=self.bucket.put_object_from_file(oss_file_path,local_file_path)
        #生成一个长期访问的oss连接地址
        url=self.bucket.sign_url(method="GET",key=oss_file_path,expires=3600*1000*24*365*100)
        print(url)
        print('http status: {0}'.format(put_oss_resualt.status))
        return url