from configs.run_config import config as run_config
from tool import process_data


if __name__ == '__main__':
    head_file_path = process_data.process_dataset(run_config['train_dataset'], 0,50000)
    training_data = process_data.load_processed_dataset(head_file_path)

    training_processed_data,training_category_data,training_news_info_data = training_data
    #import_data = process_data.import_processed_data(run_config['train_data_processed'])
    #training_processed_data, training_category_data, training_news_info_data = import_data
    print(len(training_processed_data),len(training_category_data))

    #process_data.load_dataset(run_config['validation_dataset'], 1)
    #process_data.load_dataset(run_config['test_dataset'], 2)