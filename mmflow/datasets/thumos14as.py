# cindris
import os
import os.path as osp

from .base_dataset import BaseDataset
from .builder import DATASETS

import zipfile


@DATASETS.register_module()
class Thumos14as(BaseDataset):
    """Sintel optical flow dataset.

    Args:
        pass_style (str): Pass style for Sintel dataset, and it has 2 options
            ['clean', 'final']. Default: 'clean'.
        scene (str, list, optional): Scene in Sintel dataset, if scene is None,
            it means collecting data in all of scene of Sintel dataset.
            Default: None.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def load_data_info(self) -> None:
        """Load data information, including file path of image1, image2 and
        optical flow."""

        self._get_data_dir()  # so that img1_dir is already defined

        img1_filenames = []
        img2_filenames = []

        def get_filenames(data_dir, data_suffix, img_idx=None):
            """Given folders (videos), return the image frames.
            Since pairs are needed for flow estimation:
            - img1: [0, ..., n-2]
            - img2: [1, ..., n-1]
            So that the pairs are (0,1), ..., (n-2, n-1)
            Args:
                data_dir (str - path): parent directory of the video directories.
                data_suffix (str): file type (.jpg)
                img_idx (1 or 2): whether to build img1 or img2.

            Returns:
                data_filenames: list
            """
            data_filenames = []
            for video_dir in data_dir:
                img_zip = zipfile.ZipFile(os.path.join(video_dir, "img.zip"), "r")
                frames = [i for i in img_zip.namelist() if i.endswith(data_suffix)]

                if img_idx == 1:
                    data_filenames += frames[:-1]
                elif img_idx == 2:
                    data_filenames += frames[1:]
                else:
                    data_filenames += frames

            return data_filenames

        img1_filenames = get_filenames(self.img1_dir, self.img1_suffix, 1)
        img2_filenames = get_filenames(self.img2_dir, self.img2_suffix, 2)

        self.load_img_info(self.data_infos, img1_filenames, img2_filenames)

    def _get_data_dir(self) -> None:
        """Get the paths for images and optical flow."""
        self.img1_suffix = ".jpg"
        self.img2_suffix = ".jpg"

        video_dirs = os.listdir(self.data_root)

        self.img1_dir = self.img2_dir = [
            osp.join(self.data_root, video_dir) for video_dir in video_dirs
        ]
