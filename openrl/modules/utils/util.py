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
    if epoch < total_num_epochs / 10:
        lr = initial_lr * (epoch / float(total_num_epochs/10))
    else:
        lr = initial_lr #* (1 - (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        if "name" in param_group:
            if param_group["name"] == "task_layer":
                param_group["lr"] = lr * 0.1
            else:
                param_group["lr"] = initial_lr
        else:
            param_group["lr"] = initial_lr


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e**2 / 2
