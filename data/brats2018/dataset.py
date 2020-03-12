# 生成 BRATS 2018 数据集
import numpy as np
import torch
import torch.utils.data as tdata
import os
import torchvision
import math
import glob
from skimage import transform


class BRATS2018Dataset(tdata.Dataset):

    def __init__(self, training_dir, img_dim, max_size=None):
        self.training_dir = training_dir
        self.img_dim = img_dim
        input_files = glob.glob(os.sep.join((training_dir, 'flair', '*.npy')), recursive=False)[:max_size]
        annotation_files = [os.sep.join((training_dir, 'ground_truth', os.path.split(x)[-1])) for x in input_files]
        # 添加这些数据所属的机构号, e.g.
        self.inputs = input_files
        self.annotations = annotation_files

    @staticmethod
    def preprocess(img, dim):
        # [H, W]
        img = transform.resize(img, (dim, dim))
        # [1, H, W]
        img = np.expand_dims(img, axis=0)
        return img

    @staticmethod
    def resize(img, dim):
        img = transform.resize(img, (dim, dim))
        return img

    def _preprocess(self, img):
        # [H, W]
        img = transform.resize(img, (self.img_dim, self.img_dim))
        # [1, H, W]
        img = np.expand_dims(img, axis=0)
        # float64 -> 32
        img = img.astype(np.float32)
        return img

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        imgi = self._preprocess(np.load(self.inputs[item]))  # [H, W]
        imga = self._preprocess(np.load(self.annotations[item]))
        return imgi, imga  # image, mask
        # worker_info = tdata.get_worker_info()
        # if worker_info is None:
        #     start = self.start
        #     end = self.end
        # else:
        #     per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
        #     worker_id = worker_info.id
        #     start = self.start + worker_id * per_worker
        #     end = min(start + per_worker, self.end)
        # # 返回一批数据
        # for data, anna in zip(self.inputs[start:end], self.annotations[start:end]):
        #     yield {
        #         'image': self.preprocess(np.load(data)),
        #         'mask': self.preprocess(np.load(anna))
        #     }


class InstitutionWiseBRATS2018Dataset(tdata.Dataset):

    def __init__(self, training_dir, img_dim, config_json, max_size=None):
        self.training_dir = training_dir
        self.img_dim = img_dim
        # input_files = glob.glob(os.sep.join((training_dir, 'flair', '*.npy')), recursive=False)[:max_size]
        # annotation_files = [os.sep.join((training_dir, 'ground_truth', os.path.split(x)[-1])) for x in input_files]
        # 添加这些数据所属的机构号, e.g.

        # 查找对应的名称
        import json
        with open(config_json) as f:
            cfg = json.load(f)
        institutions_its_sample_index = dict()
        images = []
        ann = []
        start_index = 0
        for institution, patient_ids in cfg.items():
            # 读取对应的 img, ann
            one_inst_ids = []
            for pid in patient_ids:
                input_files = glob.glob(os.sep.join((training_dir, 'flair', f'{pid}*.npy')), recursive=False)[
                              :max_size]
                annotation_files = [os.sep.join((training_dir, 'ground_truth', os.path.split(x)[-1])) for x in
                                    input_files]
                one_inst_ids.extend(range(start_index, start_index + len(input_files)))
                start_index = start_index + len(input_files)
                # 添加到总体的样本上去
                images.extend(input_files)
                ann.extend(annotation_files)
            institutions_its_sample_index[institution] = set(one_inst_ids)
        self.inputs = images
        self.annotations = ann
        self.institutions = cfg.keys()
        self.institutions_its_sample_index = institutions_its_sample_index

    @staticmethod
    def preprocess(img, dim):
        # [H, W]
        img = transform.resize(img, (dim, dim))
        # [1, H, W]
        img = np.expand_dims(img, axis=0)
        return img

    @staticmethod
    def resize(img, dim):
        img = transform.resize(img, (dim, dim))
        return img

    def _preprocess(self, img):
        # [H, W]
        img = transform.resize(img, (self.img_dim, self.img_dim))
        # [1, H, W]
        img = np.expand_dims(img, axis=0)
        # float64 -> 32
        img = img.astype(np.float32)
        return img

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        imgi = self._preprocess(np.load(self.inputs[item]))  # [H, W]
        imga = self._preprocess(np.load(self.annotations[item]))
        return imgi, imga  # image, mask


class InstitutionWiseBRATS2018DatasetDataAugmentation(InstitutionWiseBRATS2018Dataset):

    def __init__(self, training_dir, img_dim, config_json, max_size=None):
        super(InstitutionWiseBRATS2018DatasetDataAugmentation, self).__init__(training_dir, img_dim, config_json, max_size)
        # 这个类不对数据进行相关的处理, 处理直接在 DATASPLIT 中

    def _preprocess(self, img):
        # [H, W]
        img = transform.resize(img, (self.img_dim, self.img_dim))
        # float64 -> float32
        img = img.astype(np.float32)
        return img


if __name__ == '__main__':
    # training_dir = os.path.sep.join(('..', 'brats2018', 'training'))
    # dataset = BRATS2018Dataset(training_dir=training_dir, img_dim=128)
    # print(list(torch.utils.data.DataLoader(dataset, num_workers=2)))
    # 测试文件命名
    dirs = os.listdir(r'D:\Projects\pytorch_brats2018_fl\data\MICCAI_BraTS_2018_Data_Training\HGG')
    institutions = set()
    institutions_its_patient = dict()
    for patient_id in dirs:
        items = patient_id.split('_')
        if items[1] not in institutions:
            institutions.add(items[1])
            institutions_its_patient[items[1]] = list()
        institutions_its_patient[items[1]].append(patient_id)
    print(f'Number of institutions: {len(institutions)}')
    for k, v in institutions_its_patient.items():
        print(f'Ins {k}: num of its patient: {len(v)}')
    # # 生成配置文件
    # import json
    # with open('hgg_config.json', 'w') as fp:
    #     json.dump(institutions_its_patient, fp, indent=0)

    # 加载测试
    # training_dir = r'D:\Projects\pytorch_brats2018_fl\data\brats2018\training'
    # ds = InstitutionWiseBRATS2018Dataset(training_dir=training_dir, img_dim=128, config_json='hgg_config.json')
    # from pprint import pprint
    # # pprint(ds.institutions_its_sample_index)
    # for k, v in ds.institutions_its_sample_index.items():
    #     print(f'Ins {k}: num of its patient: {len(v)}')