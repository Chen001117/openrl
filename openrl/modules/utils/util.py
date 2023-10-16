import math


def get_grad_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    warmup_episode = 20 # TODO: change the name of the function
    if epoch < warmup_episode:
        lr = (epoch+1) / warmup_episode * initial_lr
    else:
        assert total_num_epochs - warmup_episode >= 1
        ratio = (epoch-warmup_episode) / (total_num_epochs-warmup_episode)
        ratio = max(0.1, 0.5*(1+math.cos(math.pi*ratio)))
        lr = ratio * initial_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("LR", lr, epoch, total_num_epochs)
    
    # linear decay
    # lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = lr


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e**2 / 2
