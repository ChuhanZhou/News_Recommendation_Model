from configs.run_config import config as run_config
from tool import process_data


if __name__ == '__main__':
    process_data.load_dataset(run_config['train_dataset'], 0)
    process_data.load_dataset(run_config['validation_dataset'], 1)
    process_data.load_dataset(run_config['test_dataset'], 2)


