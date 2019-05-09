#import torchvision.models as models
#(8*16)
from src.models.segment import *
from src.models.recon import *
from src.models.depth import *
from src.models.resnet import *
#
def get_model(name, n_classes, version=None):
    model = _get_model_instance(name)

    if name =='resnet' or name =='resnet_L':
        model=model()

    elif name == 'segment' or name == 'segment_L':
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=3,
                      is_deconv=True)

    elif name == 'reconstruct' or name == 'reconstruct_L':
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=3,
                      is_deconv=True)

    elif name =='depth' or name =='depth_L':
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=3,
                      is_deconv=True)

    else:
        model = model(n_classes=n_classes)

    return model

def _get_model_instance(name):
    try:
        return {
            'segment':segment,
            'resnet':resnet,
            'reconstruct':recon,
            'depth':depth,
            #Larger resolution 16*32
            'resnet_L':resnet_L,
            'segment_L':segment_L,
            'reconstruct_L':recon_L,
            'depth_L':depth_L,
        }[name]
    except:
        print('Model {} not available'.format(name))
