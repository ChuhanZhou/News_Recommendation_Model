import torch

config = {
    'word2vec_data':"C:/datasets/read/Ekstra_Bladet_word2vec/Ekstra_Bladet_word2vec/document_vector.parquet",

    'train_dataset': "C:/datasets/read/ebnerd_demo",
    #'train_dataset': "C:/datasets/read/ebnerd_large",
    'train_data_processed': "",
    'validation_dataset': "C:/datasets/read/ebnerd_demo",
    #'validation_dataset': "C:/datasets/read/ebnerd_large",
    'validation_processed': "",
    'test_dataset': "C:/datasets/read/ebnerd_testset/ebnerd_testset",
    'test_data_processed': "",

    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'lr': 4e-4,
    'epochs': 10,
    'batch_size': 8,


}
