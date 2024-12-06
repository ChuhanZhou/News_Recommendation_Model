import torch

config = {
    'word2vec_data':"C:/datasets/read/Ekstra_Bladet_word2vec/document_vector.parquet",
    'image_embeddings_data':"C:/datasets/read/Ekstra_Bladet_image_embeddings/image_embeddings.parquet",

    #'processed_data_path':'./dataset/',
    'processed_data_path':'C:/datasets/processed/',
    'train_data_processed': "ebnerd_large_validation_batch",
    'validation_data_processed': "ebnerd_demo_train_full_batch",
    'test_data_processed': "ebnerd_testset_test_full_batch",

    'thread_num': 16,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'lr': 1e-3,
    'epochs': 10,
    'batch_size': 256,#5120

    'ckpt_save_path': "./ckpt/",
    'output_path': "./out/",
}
