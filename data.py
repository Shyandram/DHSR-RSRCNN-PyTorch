import os
import torch
from PIL import Image
import glob
import random
import queue
import threading
from torch.utils.data import Dataset, DataLoader


class HazeDataset(torch.utils.data.Dataset):
    def __init__(self, ori_root, haze_root, transforms):
        self.haze_root = haze_root
        self.ori_root = ori_root
        self.image_name_list = glob.glob(os.path.join(self.haze_root, '*.jpg'))
        self.matching_dict = {}
        self.file_list = []
        self.get_image_pair_list()
        self.transforms = transforms
        print("Total data examples:", len(self.file_list))

    def __getitem__(self, item):
        """
        :param item:
        :return: haze_img, ori_img
        """
        ori_image_name, haze_image_name = self.file_list[item]
        ori_image = self.transforms(Image.open(ori_image_name))
        haze_image = self.transforms(Image.open(haze_image_name))
        return ori_image, haze_image

    def __len__(self):
        return len(self.file_list)

    def get_image_pair_list(self):
        for image in self.image_name_list:
            image = image.split("/")[-1]
            key = image.split("_")[0] + "_" + image.split("_")[1] + ".jpg"
            if key in self.matching_dict.keys():
                self.matching_dict[key].append(image)
            else:
                self.matching_dict[key] = []
                self.matching_dict[key].append(image)

        for key in list(self.matching_dict.keys()):
            for hazy_image in self.matching_dict[key]:
                self.file_list.append([os.path.join(self.ori_root, key), os.path.join(self.haze_root, hazy_image)])

        random.shuffle(self.file_list)

class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
