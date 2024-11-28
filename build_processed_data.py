from configs.run_config import config as run_config
from tool import process_data


if __name__ == '__main__':
    process_data.process_dataset("C:/datasets/read/ebnerd_demo", 0, 50000, True)
    process_data.process_dataset("C:/datasets/read/ebnerd_demo", 1, 50000, False)
    process_data.process_dataset("C:/datasets/read/ebnerd_small", 0, 50000, True)
    process_data.process_dataset("C:/datasets/read/ebnerd_small", 1, 50000, False)

    #process_data.process_dataset("C:/datasets/read/ebnerd_demo", 0, 50000, False)
    #process_data.process_dataset("C:/datasets/read/ebnerd_demo", 1, 50000, True)
    #process_data.process_dataset("C:/datasets/read/ebnerd_small", 0, 50000, False)
    #process_data.process_dataset("C:/datasets/read/ebnerd_small", 1, 50000, True)

    process_data.process_dataset("C:/datasets/read/ebnerd_large", 0, 50000, True)
    #process_data.process_dataset("C:/datasets/read/ebnerd_large", 1, 50000, True)