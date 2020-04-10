from torchvision import transforms
from .transform_ops import my_transforms_registry

def dispatcher(transforms_cfg):
    '''
    Dispatcher for Image transformation. Note that all images accepted by
    generated ops must be PIL images.

    Args
        transforms_cfg: Config node for transforms
            e.g. _C.DATASET.TRANSFORM.TRAIN.transforms
    
    Return
        a list of initialized (parameterized) image transform operators
        that can be applied to PIL image in sequence
    '''
    op_list = []
    for trans_name in transforms_cfg.transforms:
        if trans_name == "normalize" or trans_name == "none":
            continue
        callable_op = my_transforms_registry[trans_name](transforms_cfg)
        op_list.append(callable_op)
    return op_list