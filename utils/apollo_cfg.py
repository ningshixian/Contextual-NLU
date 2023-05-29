#!/usr/bin/python3  
# -*- coding: utf-8 -*-
# @Time    : 2021/6/2 14:07
# @Author  : wanghao27
# @Site    : 
# @File    : apollo_cfg.py
# @Email   : wanghao27@longfor.com
import json
import threading
import time

# import eventlet
import  requests


from utils import myconf


class ApolloCfg():
    def __init__(self,env,**kwargs):
        configs=myconf()
        configs.read("./config/env_config.ini",encoding="utf-8")
        self._apollo_server_url=configs["%s_apollo"%env]["server_url"]
        self._apollo_appid=configs["%s_apollo"%env]["appid"]
        self._apollo_cluster_name=configs["%s_apollo"%env]["cluster_name"]
        self._apollo_namespace=kwargs["namespace"]
        self._apollo_token=configs["%s_apollo"%env]["token"]
        self._apollo_api_key=configs["%s_apollo"%env]["api_key"]
        self._apollo_decrypt_url=configs["%s_apollo"%env]["decrypt_url"]
        self.decrypt_fields=kwargs["decrypt_fields"]
        self.config_cach={}
        self._notification_map={}
        self.headers={"Connection":"close"}
        self._stop=False
        for namespace in self._apollo_namespace:
            self._notification_map.update({namespace: -1})
            self.config_cach.update({namespace: {}})
        with open("./config/apollo_%s_private_key"%env,"r") as f:
            self._apollo_private_key=f.read()
        self._heart_url=("{server_url}/notifications/v2".format(server_url=self._apollo_server_url))
        self.timeout=60
        self.config_update()
        self.start_listen()

    @property
    def config(self):
        return self.config_cach
    @config.setter
    def config(self,register_message):
        print(register_message)
        if register_message["yn"]==1:
            register_message.pop("yn")
            service_name=list(register_message.keys())[0]
            # print(register_message[service_name])
            if service_name in self.config_cach["algo_gateway_ali_config"]["server_register"].keys():
                self.config_cach["algo_gateway_ali_config"]["server_register"][service_name].update(register_message[service_name])
            else:
                self.config_cach["algo_gateway_ali_config"]["server_register"].update({service_name:register_message[service_name]})
        else:
            register_message.pop("yn")
            service_name=list(register_message.keys())[0]
            # print(register_message[service_name])
            if service_name in self.config_cach["algo_gateway_ali_config"]["server_register"].keys():
                if self.config_cach["algo_gateway_ali_config"]["server_register"][service_name]== register_message:
                    self.config_cach["algo_gateway_ali_config"]["server_register"].pop(service_name)
                else:
                    if list(register_message[service_name].keys())[0] in self.config_cach["algo_gateway_ali_config"]["server_register"][service_name].keys():
                        self.config_cach["algo_gateway_ali_config"]["server_register"][service_name].pop(list(register_message[service_name].keys())[0])
    def getApolloResultCached(self,_apollo_server_url,_apollo_appid,_apollo_cluster_name,_apollo_token,_apollo_namespace):
        _cached_url=("{server_url}/configfiles/json/{appId}/{clusterName}+{token}/"
                          "{namespaceName}".format(server_url=_apollo_server_url,
                                                   appId=_apollo_appid,
                                                   clusterName=_apollo_cluster_name,
                                                   token=_apollo_token,
                                                   namespaceName=_apollo_namespace
                                                   ))
        res=requests.get(url=_cached_url)
        config_message=res.json()
        config_message["algo_gateway_ali_config"]["server_register"]=json.load(config_message["algo_gateway_ali_config"]["server_register"])
        for field in config_message:
            if field in self.decrypt_fields:
                config_message[field]=self.decrypt(config_message[field])
        return config_message

    def getApolloResult(self,_apollo_server_url,_apollo_appid,_apollo_cluster_name,_apollo_token,_apollo_namespace):
        self._url=("{server_url}/configs/{appId}/{clusterName}+{token}/"
                   "{namespaceName}".format(server_url=_apollo_server_url,
                                            appId=_apollo_appid,
                                            clusterName=_apollo_cluster_name,
                                            token=_apollo_token,
                                            namespaceName=_apollo_namespace
                                            ))
        print(self._url)
        res=requests.get(url=self._url)
        config_message=res.json()['configurations']
        for field in config_message:
            if field in self.decrypt_fields:
                config_message[field]=self.decrypt(config_message[field])
        # print(config_message)
        if "server_register" in config_message.keys():
            config_message["server_register"]=json.loads(config_message["server_register"])
        return config_message

    def start_listen(self):
        # eventlet.monkey_patch()
        # eventlet.spawn(self._heart_listen)
        threading.Thread(target=self._heart_listen).start()
    def _heart_listen(self):
        while not self._stop:
            self.config_update()
    def config_update(self):
        notifications = []
        for key in self._notification_map:
            notification_id = self._notification_map[key]
            notifications.append({
                'namespaceName': key,
                'notificationId': notification_id
            })
        self.listen_params={
            'appId': self._apollo_appid,
            'cluster': self._apollo_cluster_name,
            'notifications': json.dumps(notifications)
        }
        # print(self.listen_params)
        try:
            # print("heart_url:",self._heart_url)
            # print("listen_params:",self.listen_params)
            r = requests.get(url=self._heart_url, params=self.listen_params, timeout=self.timeout,headers=self.headers)
            # print(r.url)
            # print(r.headers)
        except requests.exceptions.ReadTimeout:
            print("No change,loop....")
        else:
            if r.status_code==200:
                configs=r.json()
                for config in configs:
                    self._notification_map[config["namespaceName"]]=config["notificationId"]
                    self.config_cach[config["namespaceName"]].update(self.getApolloResult(self._apollo_server_url,self._apollo_appid,self._apollo_cluster_name,self._apollo_token,config["namespaceName"]))
                    # print(self.config_cach)
            else:
                print(r.text)
    def decrypt(self,field):
        headers={
            "Content-Type": "application/json",
            "X-Gaia-API-Key": self._apollo_api_key,
            "Connection":"close"
        }
        body={
            "privateKey":self._apollo_private_key,
            "cipherText":[field]
        }
        res=requests.post(url=self._apollo_decrypt_url, headers=headers, data=json.dumps(body))
        return res.json()[0]


if __name__ == '__main__':
    apollo=ApolloCfg("test",decrypt_fields=["db.pass_word"],namespace=["algo_gateway_ali_config","wanghao27_config"])
