from configs.run_config import config as run_config
from tool import process_data_v2


if __name__ == '__main__':
    process_data_v2.process_dataset("C:/datasets/read/ebnerd_demo", 0)
    process_data_v2.process_dataset("C:/datasets/read/ebnerd_demo", 1)
    process_data_v2.process_dataset("C:/datasets/read/ebnerd_small", 0)
    process_data_v2.process_dataset("C:/datasets/read/ebnerd_small", 1)