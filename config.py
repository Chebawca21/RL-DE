import json
import opfunu

def getConfig():
    with open("config.json") as f:
        config = json.load(f)
    return config

def get_model_parameters(model_name, D, func_name):
    config = getConfig()
    model = config['models'][model_name]
    if 'max_fes' in model:
        model['max_fes'] = config['max_fes'][str(D)]
    model['dimension'] = D
    funcs = opfunu.get_functions_by_classname(func_name)
    func = funcs[0](ndim=D)
    model['func']= func
    return model

def get_cec_funcs(year):
    config = getConfig()
    return config['cec'][str(year)]