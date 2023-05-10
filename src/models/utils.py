import torch

from src.models.AbstractModel import AbstractModel
from src.models.SimpleTransformerModel import create_simple_transformer_model


def get_model(config: dict) -> AbstractModel:
    model_type = config['model_type']
    if model_type == 'simple_transformer':
        model = create_simple_transformer_model(config)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    return model


def save_model(config: dict, model: AbstractModel, path: str):
    '''
    Saves the model
    :param config: The config dict
    :param model: The model
    :param path: The path to save the model
    :return:
    '''
    saved_model = {
        'config': config,
        'model_state_dict': model.state_dict(),
    }

    print('saving_model', path)

    torch.save(saved_model, path)


def load_model(path: str) -> AbstractModel:
    '''
        Loads the model from the path
    '''
    saved_model = torch.load(path, map_location=torch.device('cpu'))
    config = saved_model['config']
    model = get_model(config)
    model.load_state_dict(saved_model['model_state_dict'])
    return model
