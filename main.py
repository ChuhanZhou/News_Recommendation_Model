from configs.run_config import config as run_config
from tool import process_data_v2


if __name__ == '__main__':
    print(len(process_data_v2.load_processed_dataset("{}{}".format(run_config['processed_data_path'], "ebnerd_demo_train"))))
    print(len(process_data_v2.load_processed_dataset("{}{}".format(run_config['processed_data_path'], "ebnerd_demo_validation"))))
    print(len(process_data_v2.load_processed_dataset("{}{}".format(run_config['processed_data_path'], "ebnerd_small_train"))))
    print(len(process_data_v2.load_processed_dataset("{}{}".format(run_config['processed_data_path'], "ebnerd_small_validation"))))

    process_data_v2.process_dataset("C:/datasets/read/ebnerd_large", 0)


