FROM it-container-images-registry-vpc.cn-beijing.cr.aliyuncs.com/bigdata-nlp/bigdata-nlp:docker_base
COPY . /speak_content_nlu
WORKDIR /speak_content_nlu
RUN ls /bin/nc
RUN . /root/.bashrc && \
    conda init bash && \
    conda activate python3.7 && \
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
#RUN pip install oss2 -i https://pypi.douban.com/simple/
#CMD ["python","multi_target_api.py"]
RUN /bin/cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo 'Asia/Shanghai' >/etc/timezone
#CMD ["conda","activate","python3.7"]
RUN ["chmod","+x","./command.sh"]
ENTRYPOINT ["./command.sh"]