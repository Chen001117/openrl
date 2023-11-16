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
    # epoch = 3200 steps
    
    # lr = 1e-3 * (epoch+1) / 6
    # lr = (1e-4-1e-5) * ((2100-epoch)/200) + 1e-5
    # lr = 1e-3 - (1e-3 * (epoch / float(total_num_epochs)))
    # lr = 1e-4 + 0.5 * (1e-3-1e-4) * (1 + math.cos((epoch-5)/(3120)*math.pi))
    
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    return lr


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e**2 / 2
