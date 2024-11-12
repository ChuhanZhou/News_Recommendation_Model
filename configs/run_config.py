import torch

config = {
    'word2vec_data':"C:/datasets/read/Ekstra_Bladet_word2vec/Ekstra_Bladet_word2vec/document_vector.parquet",

    'train_dataset': "C:/datasets/read/ebnerd_demo",
    #'train_dataset': "C:/datasets/read/ebnerd_large",
    'train_data_processed': "./dataset/ebnerd_demo.train",
    'validation_dataset': "C:/datasets/read/ebnerd_demo",
    #'validation_dataset': "C:/datasets/read/ebnerd_large",
    'validation_processed': "./dataset/ebnerd_demo.validation",
    'test_dataset': "C:/datasets/read/ebnerd_testset/ebnerd_testset",
    'test_data_processed': "./dataset/ebnerd_testset.test",

    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'lr': 1e-4,
    'epochs': 10,
    'batch_size': 1024,


}
