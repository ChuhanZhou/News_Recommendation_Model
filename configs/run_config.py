import torch

config = {
    'word2vec_data':"C:/datasets/read/Ekstra_Bladet_word2vec/document_vector.parquet",
    'image_embeddings_data':"C:/datasets/read/Ekstra_Bladet_image_embeddings/image_embeddings.parquet",

    'processed_data_path':'./dataset/',
    'train_data_processed': "ebnerd_small_train_batch",
    'validation_data_processed': "ebnerd_demo_validation",
    'test_data_processed': "ebnerd_testset_test",

    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'lr': 1e-3,
    'epochs': 50,
    'batch_size': 1024,#5120

    'ckpt_save_path': "./ckpt/",
    'test_ckpt_path': "./ckpt/ckpt_ebnerd_demo_final.pth",
}
