from configparser import ConfigParser


class myconf(ConfigParser):
    def __init__(self, defaults=None):
        ConfigParser.__init__(self, defaults=defaults)

        # 这里重写了optionxform方法，直接返回选项名
    def optionxform(self, optionstr):
        return optionstr