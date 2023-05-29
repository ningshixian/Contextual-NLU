import re


def force_no_escaping(x):
    x = x.replace("\\", "\\\\").replace("|", "\\"+"|")
    for s in r"*.?+$^()[]{}":    # raw string，不转义
        x = x.replace(s, "\\"+s)
    return x


def clean(text):
    pattern = r"(.*有什么可以帮.*|.*有什么可以.*服务.*|.*有什么可以.*效劳.*|.*为您服务|感谢您的咨询.*|.*人工.*|.*服务.*评价.*|欢迎回来|.*继续为您服务.*|.*您是否还在线.*)"
    text = re.sub(pattern, '', str(text))
    pattern = r"(^你好.?$|^您好.?$|^开了.?$|^不?是.?$|^不?对.?$|^在.?$|^在么.?$|^嗯.?$|^？$|^没有.?$)"
    text = re.sub(pattern, '', str(text))

    # 删掉<>标签
    text = re.sub('<[^<]+?>', '', str(text))
    # 去除不可见字符
    text = re.sub(
        "[\001\002\003\004\005\006\007\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a]+","",text,
    )
    # 图片替换为[PIC]
    text = re.sub('(<img[^<]+?>)|(http[s]?://(.*)image(.*))', '[PIC]', str(text))
    # http替换为[HTTP]
    text = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", '[HTTP]', text).replace('\t','').replace('&nbsp','').strip()
    # # 时间替换为[TIME]
    # text = time_ext(text)
    # JSON字符串替换为[JSON]
    text = re.sub(r'\{.*\}', '[JSON]', str(text))
    # 手机号码替换为[PHONE]
    pattern = r"(?:^|[^\d])((?:\+?86)?1(?:3\d{3}|5[^4\D]\d{2}|8\d{3}|7(?:[01356789]\d{2}|4(?:0\d|1[0-2]|9\d))|9[189]\d{2}|6[567]\d{2}|4(?:[14]0\d{1}|[68]\d{2}|[579]\d{2}))\d{6})(?:$|[^\d])"
    phone_list = re.compile(pattern).findall(text)
    # phone_list.append("+8618646601608")
    for phone in phone_list:
        phone = force_no_escaping(phone)
        text = re.sub(phone, '[PHONE]', text)
    # text = re.sub(r"(?:^|\D)?(?:\+?86)?1(?:3\d{3}|5[^4\D]\d{2}|8\d{3}|7(?:[01356789]\d{2}|4(?:0\d|1[0-2]|9\d))|9[189]\d{2}|6[567]\d{2}|4(?:[14]0\d{1}|[68]\d{2}|[579]\d{2}))\d{6}(?:^|\D)?", '[PHONE]', text)
    text = re.sub("1(\d{2})((\*){4})(\d{4})", '[PHONE]', text)  # 135****4934
    # 尾号替换为[SUBPN]
    text = re.sub("尾号.?(\d{4})", '尾号[SUBPN]', text)
    text = re.sub("(\d{4}).?尾号", '[SUBPN]尾号', text)

    text = text.replace("PIC", "[PIC]").replace("HTTP", "[HTTP]").replace("TIME", "[TIME]").replace("PHONE", "[PHONE]").replace("SUBPN", "[SUBPN]").replace("[[", "[").replace("]]", "]")
    return text


# 坐席对话数据噪声较多，降噪
def denoise():
    pass


