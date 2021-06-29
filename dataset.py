import torch
import torch.distributed as dist
from torch.utils.data import Sampler

class DictTensorDataset(torch.utils.data.Dataset):

    def __init__(self, dt):
        self.length = list(dt.values())[0].size(0)
        assert all(self.length == tensor.size(0) for tensor in dt.values())
        self.dt = dt

    def __getitem__(self, index):
        return dict([(key, self.dt[key][index]) for key in self.dt.keys()])

    def __len__(self):
        return self.length

class DistributedBatchSampler(Sampler):
    """
    This distrubuted sampler accepts the number of batches as input and
    sample the corresponding batches in a distributed way
    """

    def __init__(self, dataset, batch_size, nsamples, num_replicas=None, rank=None, shuffle=True, seed=58):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.ndataset = len(dataset)
        self.batch_size = batch_size
        self.nsamples = nsamples
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        batches = []
        pool = []
        offset = self.rank
        while len(batches) < self.nsamples:
            if self.shuffle:
                indices = torch.randperm(self.ndataset, generator=g).tolist()
            else:
                indices = list(range(self.ndataset))
            indices = indices[offset::self.num_replicas]
            offset = (offset + len(indices) * self.num_replicas) % self.ndataset
            
            pool += indices
            while len(batches) < self.nsamples and len(pool) >= self.batch_size:
                batches.append(pool[:self.batch_size])
                pool = pool[self.batch_size:]

        return iter(batches)

    def __len__(self):
        return self.nsamples

if __name__ == '__main__':
    class Data():
        def __len__(self):
            return 20
    
    dataset = Data()
    bsamplers = []
    for i in range(5):
        bsamplers.append(DistributedBatchSampler(dataset, 4, 10, num_replicas=5, rank=i))

    iters = [iter(bsampler) for bsampler in bsamplers]
    for step in range(10):
        print('step = {}'.format(step + 1))
        for it in iters:
            print(next(it))


