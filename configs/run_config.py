import torch

config = {
    'word2vec_data':"C:/datasets/read/Ekstra_Bladet_word2vec/Ekstra_Bladet_word2vec/document_vector.parquet",

    'train_dataset': "C:/datasets/read/ebnerd_demo",
    'train_data_processed': "./dataset/ebnerd_demo.train",
    #'train_dataset': "C:/datasets/read/ebnerd_large",
    #'train_data_processed': "./dataset/ebnerd_large.train",

    'validation_dataset': "C:/datasets/read/ebnerd_demo",
    'validation_data_processed': "./dataset/ebnerd_demo.validation",
    #'validation_dataset': "C:/datasets/read/ebnerd_large",
    #'validation_processed': "./dataset/ebnerd_large.validation",

    'test_dataset': "C:/datasets/read/ebnerd_testset/ebnerd_testset",
    'test_data_processed': "./dataset/ebnerd_testset.test",

    'neg_label_max_num':10,

    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'lr': 3e-4,
    'epochs': 50,
    'batch_size': 256,

    'ckpt_save_path': "./ckpt/",
    'test_ckpt_path': "./ckpt/ckpt_ebnerd_demo_final.pth",
    #'test_ckpt_path': "./ckpt/ckpt_ebnerd_demo_epoch_3.pth",
    #'test_ckpt_path': "./ckpt/ckpt_ebnerd_large_epoch_4.pth",
}
