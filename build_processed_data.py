from configs.run_config import config as run_config
from tool import process_data

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process data")
    parser.add_argument('--path', help="dataset path",default=None)
    parser.add_argument('--type', help="data type:[0]train [1]validation [2]test", type=int)
    parser.add_argument('--sub', help="subvolume size", type=int, default=10000)
    parser.add_argument('--batch', help="batch type:[0] data only for training,[1] full data can't run in batch,[2] full data can run in batch", type=int)
    parser.add_argument('--out', help="result output path", default=run_config['processed_data_path'])
    parser.add_argument('--thread', help="how may thread can be used", type=int, default=run_config['thread_num'])
    args = parser.parse_args()

    run_config['processed_data_path'] = args.out
    run_config['thread_num'] = args.thread

    if args.path is not None:
        process_data.process_dataset(args.path, type_i=args.type, subvolume_item_num=args.sub,batch_type=args.batch)
    else:
        process_data.process_dataset("C:/datasets/read/ebnerd_testset", type_i=2, subvolume_item_num=20000, batch_type=2)

        #process_data.process_dataset("C:/datasets/read/ebnerd_demo", type_i=0, subvolume_item_num=50000, batch_type=0)
        #process_data.process_dataset("C:/datasets/read/ebnerd_demo", type_i=1, subvolume_item_num=50000, batch_type=0)
        #process_data.process_dataset("C:/datasets/read/ebnerd_demo", type_i=0, subvolume_item_num=50000, batch_type=2)
        #process_data.process_dataset("C:/datasets/read/ebnerd_demo", type_i=1, subvolume_item_num=50000, batch_type=2)
        #process_data.process_dataset("C:/datasets/read/ebnerd_small", type_i=0, subvolume_item_num=50000, batch_type=2)
        #process_data.process_dataset("C:/datasets/read/ebnerd_small", type_i=1, subvolume_item_num=50000, batch_type=2)

        #process_data.process_dataset("C:/datasets/read/ebnerd_large", type_i=0, subvolume_item_num=50000, batch_type=0)
        #process_data.process_dataset("C:/datasets/read/ebnerd_large", type_i=1, subvolume_item_num=50000, batch_type=0)
