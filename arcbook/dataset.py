import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import mxnet as mx
import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn
from torchvision.transforms import InterpolationMode


from PIL import Image
import random



def get_dataloader(
    root_dir,
    local_rank,
    batch_size,
    dali = False,
    dali_aug = False,
    seed = 2048,
    num_workers = 2,
    ) -> Iterable:

    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None


    # Mxnet RecordIO
    if os.path.exists(rec) and os.path.exists(idx):
        print("transform start")
        train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)

    # DALI
    if dali:
        return dali_data_iter(
            batch_size=batch_size, rec_file=rec, idx_file=idx,
            num_threads=2, local_rank=local_rank, dali_aug=dali_aug)

    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self
class ThreePartImageTransform:
    def __call__(self, image):
        # 이미지를 PIL Image에서 NumPy 배열로 변환
        image = np.array(image)

        # 이미지 크기 얻기
        height, width = image.shape[:2]

        # 세로 크기 3등분
        s_height = height // 3
        if height % 3 != 0:
            s_height += 1

        # 세 부분 나누기
        first_part = image[:s_height, :]
        second_part = image[s_height:s_height * 2, :] if s_height * 2 < height else image[s_height:, :]
        third_part = image[s_height * 2:, :] if s_height * 2 < height else np.zeros_like(image[:s_height, :])

        # 가장 큰 높이 계산
        max_height = max(first_part.shape[0], second_part.shape[0], third_part.shape[0])

        # 패딩을 위한 함수 정의
        def pad_height(part, max_height):
            if part.shape[0] < max_height:
                pad_height = max_height - part.shape[0]
                pad = np.zeros((pad_height, width, 3), dtype=part.dtype)
                part = np.vstack([part, pad])
            return part

        # 각 부분에 대해 패딩 적용
        first = pad_height(first_part, max_height)
        second = pad_height(second_part, max_height)
        third = pad_height(third_part, max_height)

        # 세 부분을 리스트로 만들어 랜덤하게 섞기
        parts = [first, second, third]
        random.shuffle(parts)

        # 섞인 부분을 수평으로 결합
        combined_image = np.hstack(parts)

        # NumPy 배열을 PIL Image로 다시 변환
        return Image.fromarray(combined_image.astype(np.uint8))


class nThreePartImageTransform:
    def __init__(self, min_n:int = 1, max_n: int=10):
        ''' 등분할 n의 범위 설정(min~max) '''
        self.min_n = min_n
        self.max_n = max_n
    def v2_find_n_for_equal_division(self, w: int, h: int):
        ''' w: width, h: height '''
        min_padding = float("inf")
        best_n = self.min_n
        # 최대 n 값을 설정. 기본값은 이미지의 높이로 설정
        max_n = self.max_n if self.max_n is not None else h
        for n in range(self.min_n, max_n + 1):
            divided_height = round(h / n)
            total_width = w * n
            total_height = divided_height
            # 필요한 padding 계산(정사각형으로 만들기 위해, 부족한 쪽에 대해 고려)
            if total_height < total_width:
                padding = (total_width - total_height) * total_width
            else:
                padding = (total_height - total_width) * total_height
            # 패딩이 최소인 n 값을 선택
            if padding < min_padding:
                min_padding = padding
                best_n = n
        return best_n
    def __call__(self, image):
        # 이미지를 PIL Image에서 NumPy 배열로 변환
        image = np.array(image)
        # 이미지 크기 얻기
        height, width = image.shape[:2]
        # 최적의 n 값 찾기
        best_n = self.v2_find_n_for_equal_division(width, height)
        # 각 부분의 높이 계산
        s_height = height // best_n
        if height % best_n != 0:
            s_height += 1
        # n 부분으로 나누기
        parts = []
        for i in range(best_n):
            start = i * s_height
            end = start + s_height if start + s_height < height else height
            parts.append(image[start:end, :])
        # 가장 큰 높이 계산
        max_height = max(part.shape[0] for part in parts)
        # 패딩을 위한 함수 정의
        def pad_height(part, max_height):
            if part.shape[0] < max_height:
                pad_height = max_height - part.shape[0]
                pad = np.zeros((pad_height, width, 3), dtype=part.dtype)
                part = np.vstack([part, pad])
            return part
        # 각 부분에 대해 패딩 적용
        padded_parts = [pad_height(part, max_height) for part in parts]
        # 부분들을 리스트로 만들어 랜덤하게 섞기
        random.shuffle(padded_parts)
        # 섞인 부분을 수평으로 결합
        combined_image = np.hstack(padded_parts)
        # NumPy 배열을 PIL Image로 다시 변환
        return Image.fromarray(combined_image.astype(np.uint8))


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             ThreePartImageTransform(),
             transforms.RandomAutocontrast(),
             transforms.ColorJitter(),
             transforms.RandomGrayscale(),
             transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             transforms.RandomRotation(degrees=30),
             transforms.RandomPerspective(distortion_scale=0.3),
             transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.9, 1.1)),
             transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             transforms.RandomErasing(),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000


def dali_data_iter(
    batch_size: int, rec_file: str, idx_file: str, num_threads: int,
    initial_fill=32768, random_shuffle=True,
    prefetch_queue_depth=1, local_rank=0, name="reader",
    mean=(127.5, 127.5, 127.5), 
    std=(127.5, 127.5, 127.5),
    dali_aug=False
    ):
    """
    Parameters:
    ----------
    initial_fill: int
        Size of the buffer that is used for shuffling. If random_shuffle is False, this parameter is ignored.

    """
    rank: int = distributed.get_rank()
    world_size: int = distributed.get_world_size()
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    def dali_random_resize(img, resize_size, image_size=112):
        img = fn.resize(img, resize_x=resize_size, resize_y=resize_size)
        img = fn.resize(img, size=(image_size, image_size))
        return img
    def dali_random_gaussian_blur(img, window_size):
        img = fn.gaussian_blur(img, window_size=window_size * 2 + 1)
        return img
    def dali_random_gray(img, prob_gray):
        saturate = fn.random.coin_flip(probability=1 - prob_gray)
        saturate = fn.cast(saturate, dtype=types.FLOAT)
        img = fn.hsv(img, saturation=saturate)
        return img
    def dali_random_hsv(img, hue, saturation):
        img = fn.hsv(img, hue=hue, saturation=saturation)
        return img
    def multiplexing(condition, true_case, false_case):
        neg_condition = condition ^ True
        return condition * true_case + neg_condition * false_case

    condition_resize = fn.random.coin_flip(probability=0.1)
    size_resize = fn.random.uniform(range=(int(112 * 0.5), int(112 * 0.8)), dtype=types.FLOAT)
    condition_blur = fn.random.coin_flip(probability=0.2)
    window_size_blur = fn.random.uniform(range=(1, 2), dtype=types.INT32)
    condition_flip = fn.random.coin_flip(probability=0.5)
    condition_hsv = fn.random.coin_flip(probability=0.2)
    hsv_hue = fn.random.uniform(range=(0., 20.), dtype=types.FLOAT)
    hsv_saturation = fn.random.uniform(range=(1., 1.2), dtype=types.FLOAT)

    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads,
        device_id=local_rank, prefetch_queue_depth=prefetch_queue_depth, )
    condition_flip = fn.random.coin_flip(probability=0.5)
    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file, index_path=idx_file, initial_fill=initial_fill, 
            num_shards=world_size, shard_id=rank,
            random_shuffle=random_shuffle, pad_last_batch=False, name=name)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        if dali_aug:
            images = fn.cast(images, dtype=types.UINT8)
            images = multiplexing(condition_resize, dali_random_resize(images, size_resize, image_size=112), images)
            images = multiplexing(condition_blur, dali_random_gaussian_blur(images, window_size_blur), images)
            images = multiplexing(condition_hsv, dali_random_hsv(images, hsv_hue, hsv_saturation), images)
            images = dali_random_gray(images, 0.1)

        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, mean=mean, std=std, mirror=condition_flip)
        pipe.set_outputs(images, labels)
    pipe.build()
    return DALIWarper(DALIClassificationIterator(pipelines=[pipe], reader_name=name, ))


@torch.no_grad()
class DALIWarper(object):
    def __init__(self, dali_iter):
        self.iter = dali_iter

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict['data'].cuda()
        tensor_label: torch.Tensor = data_dict['label'].cuda().long()
        tensor_label.squeeze_()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def reset(self):
        self.iter.reset()
