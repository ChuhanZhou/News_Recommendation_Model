from configs.run_config import config as run_config
from tool import process_data

if __name__ == '__main__':


    training_data = process_data.load_processed_dataset(run_config['train_data_processed'])

    training_processed_data,training_category_data,training_news_info_data = training_data

    print(len(training_processed_data),len(training_category_data))