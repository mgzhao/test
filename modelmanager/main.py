import argparse
import os

import modelmanager.config as config
import modelmanager.distributed as distributed
from modelmanager.pbt_wrapper import PbtModel


def run(models, config_path=None, pbt_path=None, basepath=None):
    if config_path is None:
        parser = argparse.ArgumentParser("Train PyTorch model")
        parser.add_argument(
            '--pbt', default='', help="Path to pbt config file")
        parser.add_argument(
            '--basepath', default='', help="Path to project root directory")
        parser.add_argument('config', help="Path to model config file")
        args = parser.parse_args()
        pbt_path = args.pbt
        basepath = args.basepath or basepath
        config_path = args.config
    model_args = config.load(config_path)
    model_name = model_args['model']
    del model_args['model']
    if "basepath" not in model_args and basepath:
        model_args["basepath"] = basepath
    Model = models[model_name]
    if pbt_path:
        import pbt
        print('Running HPO on model with parameters {}'.format(model_args))
        model = PbtModel(Model, model_args)
        pbt.run(model, pbt_path)
    else:
        print('Running model with parameters {}'.format(model_args))
        if "distributed" in model_args:
            distributed.train(Model, model_args)
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
                map(str, model_args['gpus']))
            model = Model(**model_args)
            model.train()
            return model
