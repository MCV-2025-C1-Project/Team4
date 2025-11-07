import os
from pathlib import Path
from typing import Callable, Self
import cv2
import numpy as np
import tqdm

from libs_week4.descriptor import KeypointAndDescriptorMaker, Scorer


class Image:
    def __init__(self, id: int, image: np.ndarray, mask: np.ndarray, author_title: str | None):
        self.id = id
        self.image = image
        self.mask = mask
        self.author_title = author_title
        self.keypoints = None
        self.descriptors = None
        self.valid = None
        self.score = None
        self.info = None

class ImageDatabase:
    def __init__(self, images: list[Image]):
        self.images = images

    def reset_descriptors_distances_and_scores(self):
        for image in self.images:
            image.keypoints = None
            image.descriptors = None
            image.valid = None
            image.score = None
            image.info = None

    def compute_keypoints_and_descriptors(self, descriptor_maker: KeypointAndDescriptorMaker):
        for image in tqdm.tqdm(self.images):
            image.keypoints, image.descriptors = descriptor_maker.detect_and_compute(image.image, image.mask)

    def compute_keypoint_descriptor_statistics(self) -> dict:
        """
        Compute statistics about keypoints and descriptors across all images in the database.

        Returns:
            dict: Statistics including count, mean, std, min, max for keypoints,
                  and descriptor dimensions.
        """
        keypoint_counts = []
        descriptor_sizes = []
        descriptor_dims = set()

        for image in self.images:
            if image.keypoints is not None:
                keypoint_counts.append(len(image.keypoints))

            if image.descriptors is not None and len(image.descriptors) > 0:
                descriptor_sizes.append(len(image.descriptors))
                descriptor_dims.add(image.descriptors.shape[1] if len(image.descriptors.shape) > 1 else 1)

        if not keypoint_counts:
            return {
                "error": "No keypoints computed yet"
            }

        stats = {
            "keypoints": {
                "count": len(keypoint_counts),
                "mean": float(np.mean(keypoint_counts)),
                "std": float(np.std(keypoint_counts)),
                "min": int(np.min(keypoint_counts)),
                "max": int(np.max(keypoint_counts)),
                "median": float(np.median(keypoint_counts)),
                "total": int(np.sum(keypoint_counts))
            },
            "descriptors": {
                "count": len(descriptor_sizes),
                "mean": float(np.mean(descriptor_sizes)) if descriptor_sizes else 0,
                "std": float(np.std(descriptor_sizes)) if descriptor_sizes else 0,
                "min": int(np.min(descriptor_sizes)) if descriptor_sizes else 0,
                "max": int(np.max(descriptor_sizes)) if descriptor_sizes else 0,
                "median": float(np.median(descriptor_sizes)) if descriptor_sizes else 0,
                "total": int(np.sum(descriptor_sizes)) if descriptor_sizes else 0,
                "dimensions": list(descriptor_dims) if descriptor_dims else []
            }
        }

        return stats

    def query(self, query_image: np.ndarray, query_keypoints, query_descriptors, scorer: Scorer, k) -> list[Image]:
        for image in self.images:
            image.valid, image.score, image.info = scorer.score(
                query_image, query_keypoints, query_descriptors,
                image.image, image.keypoints, image.descriptors,
            )

        self.images.sort(key=lambda im: im.score, reverse=True)

        result = []
        for image in self.images:
            if len(result) >= k:
                break
            if image.valid:
                result.append(image)

        if not result:
            return []

        return result


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
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Could not read mask {mask_path.name}.")
            else:
                mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
            
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
