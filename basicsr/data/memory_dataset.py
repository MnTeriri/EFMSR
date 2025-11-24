from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import bgr2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MemoryDataset(data.Dataset):
    def __init__(self, opt):
        super(MemoryDataset, self).__init__()
        self.opt = opt
        self.io_backend_opt = opt['io_backend']
        self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)  # 用于加载文件
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        # 获取标准质量（opt['dataroot_gt']）和低质量图像（opt['dataroot_lq']）所在文件夹路径
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']

        # 获取低质量图像文件名模板，例如'{}'或'{}x2'
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        # 获取图片路径，格式：[{'lq_path': ..., 'gt_path': ...}, ...]
        self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
        self.images = []

        for path in self.paths:
            gt_path = path['gt_path']
            lq_path = path['lq_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            self.images.append((img_gt, img_lq))

    def __getitem__(self, index):
        scale = self.opt['scale']
        # 用于训练的数据增强
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        img_gt, img_lq = self.images[index]
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # 随机裁剪
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # 翻转、旋转
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # 颜色空间转换
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # 在验证或测试阶段裁剪尺寸不匹配的 GT 图像，尤其是针对 SR 基准数据集
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR 转 RGB，HWC 转 CHW，NumPy 数组转为张量
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
