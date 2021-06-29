from utils import dotdict

def get_learning_rate(it):
    return 2e-4

config = dotdict({
    'num_features': 64,
    'num_res_blocks': 8,

    'workers': 4,
    'total_iter': 10000,
    'batch_size_per_gpu': 32,
    'master_port': 17937,
    'weight_decay': 1e-4,
    'num_embeddings': 6,
    'print_freq': 100,
    'train_steps': 1000,
    'get_learning_rate': get_learning_rate,
    'script_batch_size': 256,
})

