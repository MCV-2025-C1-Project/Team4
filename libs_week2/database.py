import os
from pathlib import Path
from typing import Callable, Self
import cv2
import numpy as np

from libs_week1.descriptor import ImageDescriptorMaker


class Image:
    def __init__(self, id: int, image: np.ndarray, mask: np.ndarray, author_title: str | None):
        self.id = id
        self.image = image
        self.mask = mask
        self.author_title = author_title
        self.descriptor = None
        self.distance = None

class ImageDatabase:
    def __init__(self, images: list[Image]):
        self.images = images

    def reset_descriptors_and_distances(self):
        for image in self.images:
            image.descriptor = None
            image.distance = None

    def compute_descriptors(self, descriptor_maker: ImageDescriptorMaker):
        for image in self.images:
            image.descriptor = descriptor_maker.make_descriptor(image.image)

    def query(self, query_descriptor: np.ndarray, distance: Callable[[np.ndarray, np.ndarray], float], k):
        for image in self.images:
            image.distance = distance(image.descriptor, query_descriptor)

        self.images.sort(key=lambda im: im.distance)

        return self.images[:k]


    @staticmethod
    def load(database_path: str) -> 'ImageDatabase':
        images: list[Image] = []
        for filename in sorted(os.listdir(database_path)):
            if not filename.endswith(".jpg"):
                continue

            image_path = os.path.join(database_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Could not read image {filename}.")
            
            mask_path = Path(os.path.join(database_path, filename)).with_suffix('.png')
            if mask_path.exists():
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask is None:
                    raise ValueError(f"Could not read mask {mask_path.name}.")
            else:
                mask = np.ones(image.shape[:2])
            
            painting_name_path = Path(image_path).with_suffix('.txt')
            try:
                author_title = painting_name_path.read_text(encoding="ISO-8859-1")
            except Exception as e:
                print(f"Image {filename} doesn't have associated .txt file.")
                author_title = None

            stem = Path(filename).stem
            id = int(stem.split('_')[1])

            images.append(Image(id, image, mask, author_title))

        return ImageDatabase(images)
