from keras.optimizers import RMSprop, SGD, Nadam
from bert4keras.optimizers import *

weight_decay = 0.01  # 权重衰减系数，类似模型正则项策略，避免模型过拟合


def get_opt(optimizer, LR):
    # 优化器选择
    if optimizer.lower() == "sgd":
        opt = SGD(lr=LR, decay=1e-5, momentum=0.9, nesterov=True)
        # opt = SGD(lr=1e-3, decay=1e-3/EPOCHS, momentum=0.9, nesterov=True)
    elif optimizer.lower() == "adam":
        opt = Adam(lr=LR, clipvalue=1.0)
    elif optimizer.lower() == "nadam":
        opt = Nadam(lr=LR, clipvalue=1.0)
    elif optimizer.lower() == "rmsprop":
        opt = RMSprop(lr=LR, clipvalue=1.0)
    elif optimizer.lower() == "adamw":
        # 变成带权重衰减的Adam
        AdamW = extend_with_weight_decay(Adam, "AdamW")
        opt = AdamW(LR, weight_decay_rate=weight_decay)
    elif optimizer.lower() == "adamlr":
        # 变成带分段线性学习率的Adam
        AdamLR = extend_with_piecewise_linear_lr(Adam, "AdamLR")
        # 实现warmup，前1000步学习率从0增加到0.001
        opt = AdamLR(learning_rate=LR, lr_schedule={
            1000: 1,
            2000: 0.1
        })
    elif optimizer.lower() == "adamga":
        # 变成带梯度累积的Adam
        AdamGA = extend_with_gradient_accumulation(Adam, "AdamGA")
        opt = AdamGA(learning_rate=LR, grad_accum_steps=10)
    elif optimizer.lower() == "adamla":
        # 变成加入look ahead的Adam
        AdamLA = extend_with_lookahead(Adam, "AdamLA")
        opt = AdamLA(learning_rate=LR, steps_per_slow_update=5, slow_step_size=0.5)
    elif optimizer.lower() == "adamlo":
        # 变成加入懒惰更新的Adam
        AdamLO = extend_with_lazy_optimization(Adam, "AdamLO")
        opt = AdamLO(learning_rate=LR, include_in_lazy_optimization=[])
    elif optimizer.lower() == "adamwlr":
        # 组合使用
        AdamW = extend_with_weight_decay(Adam, "AdamW")
        AdamWLR = extend_with_piecewise_linear_lr(AdamW, "AdamWLR")
        # 带权重衰减和warmup的优化器
        opt = AdamWLR(
            learning_rate=LR, weight_decay_rate=weight_decay, lr_schedule={1000: 1.0}
        )
    elif optimizer.lower() == "adamema":
        AdamEMA = extend_with_exponential_moving_average(Adam, name="AdamEMA")
        opt = AdamEMA(LR, ema_momentum=0.9999)
    return opt
