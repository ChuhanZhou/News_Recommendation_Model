from configs.run_config import config as run_config
from tool import process_data


if __name__ == '__main__':

    process_data.process_dataset(run_config['validation_dataset'], 1,50000)
    #process_data.load_dataset(run_config['test_dataset'], 2)


