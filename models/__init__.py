import torchvision.models as models
from models.esrcnn_s2self import *
from models.esrcnn_s2l8_1 import *
from models.esrcnn_s2l8_2 import *
from models.esrcnn_s2l8_3 import *

def get_model(name,opt):
    if name == 'esrcnn_s2self':
        model = esrcnn_s2self(opt)
    elif name == 'esrcnn_s2l8_1':
    	model = esrcnn_s2l8_1(opt)
    elif name == 'esrcnn_s2l8_2':
    	model = esrcnn_s2l8_2(opt)
    elif name == 'esrcnn_s2l8_3':
    	model = esrcnn_s2l8_3(opt)

    return model
