import os
import logging

class dotdict(dict):
    def __setattr__(self, name, value):
        self[name] = value

    def __getattr__(self, name):
        if name in self:
            return self[name]
        return super().__getattr__(name)

def ensure_dir(dir_name):
    os.makedirs(dir_name, exist_ok=True)

def get_train_log_dir():
    work_dir = os.getcwd()
    config_dir = os.path.dirname(work_dir)

    assert os.path.basename(config_dir) == 'config', \
           "Working directory should be under 'config'..."

    base_dir = os.path.dirname(config_dir)
    exp_name = os.path.basename(work_dir)
    train_log_dir = os.path.join(base_dir, 'train_log', exp_name)
    ensure_dir(train_log_dir)

    link_dir = os.path.join(work_dir, 'train_log')
    if os.path.islink(link_dir):
        os.unlink(link_dir)
    assert not os.path.exists(link_dir), \
           "'train_log' should be reserved for soft link"
    os.symlink(train_log_dir, link_dir)
    return train_log_dir

def init_logger(train_log_path):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(train_log_path, 'log.txt'), mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s[%(levelname)s]: %(message)s'))
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def to_string(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

