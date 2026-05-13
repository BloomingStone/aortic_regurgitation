from pathlib import Path

from dataclasses import dataclass
from typing import Protocol
from collections import namedtuple

import numpy as np
import torch
import nibabel as nib


@dataclass
class MedicalImage:
    data: torch.Tensor
    affine: np.ndarray
    
    def to_device(self, device: torch.device) -> 'MedicalImage':
        return MedicalImage(
            data=self.data.to(device),
            affine=self.affine
        )
    
    def to_nii(self) -> nib.Nifti1Image:
        return nib.Nifti1Image(self.data.cpu().numpy().squeeze(), self.affine)
    
    def save_nii(self, path: Path):
        nib.save(self.to_nii(), path)


class Transform(Protocol):
    def __call__(self, image: MedicalImage) -> MedicalImage:
        return self.transform_image(image)
    
    def transform_image(self, image: MedicalImage) -> MedicalImage:
        return MedicalImage(
            data=self.transform_data(image.data),
            affine=self.transform_affine(image.affine)
        )
    
    def transform_affine(self, affine: np.ndarray) -> np.ndarray:
        ...
    
    def transform_data(self, data: torch.Tensor) -> torch.Tensor:
        ...

class IdentityTransform(Transform):
    def transform_image(self, image: MedicalImage) -> MedicalImage:
        return image
    
    def transform_affine(self, affine: np.ndarray) -> np.ndarray:
        return affine


AABB = namedtuple('AABB', ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'])


class ClipROITransform(Transform):
    def __init__(self, roi: AABB):
        self.roi = roi
    
    def transform_data(self, data: torch.Tensor) -> torch.Tensor:
        x_min, x_max, y_min, y_max, z_min, z_max = self.roi
        return data[..., x_min:x_max, y_min:y_max, z_min:z_max]
    
    def transform_affine(self, affine: np.ndarray) -> np.ndarray:
        x_min, y_min, z_min = self.roi.x_min, self.roi.y_min, self.roi.z_min
        new_origin_world = np.array([x_min, y_min, z_min, 1]) @ affine.T
        new_affine = affine.copy()
        new_affine[:3, 3] = new_origin_world[:3]
        return new_affine

class ResampleTransform(Transform):
    def __init__(self, resample_factor: float | np.ndarray):
        if isinstance(resample_factor, (int, float)):
            self.resample_factor = np.array([resample_factor] * 3)
        else:
            self.resample_factor = np.array(resample_factor)
        
        assert np.all(self.resample_factor > 0), "Resample factor must be positive"
        assert self.resample_factor.shape == (3,), "Resample factor must be a 3D array"

    def transform_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Resample the last three dimensions (spatial axes) of the input tensor.
        Supports any leading dimensions (batch, channel, etc.).
        """
        if data.ndim < 3:
            raise ValueError(f"Data must have at least 3 spatial dimensions, got {data.ndim}")
        
        if data.dtype in [torch.int16, torch.int32, torch.int64, torch.uint8]:
            mode = 'nearest'
        else:
            mode = 'trilinear'

        original_shape = data.shape
        spatial_shape = original_shape[-3:]          # (D, H, W)
        leading_shape = original_shape[:-3]          # all non-spatial dims

        # Flatten leading dims into a single batch dimension, add a dummy channel
        # Shape becomes (B, 1, D, H, W) where B = product(leading_shape)
        data_flat = data.view(-1, 1, *spatial_shape)

        # Apply trilinear interpolation
        resampled = torch.nn.functional.interpolate(
            data_flat,
            scale_factor=self.resample_factor.tolist(),
            mode=mode,
        )

        # New spatial dimensions after resampling
        new_spatial_shape = resampled.shape[2:]      # (newD, newH, newW)
        # Restore original leading dimensions + new spatial shape
        new_shape = leading_shape + new_spatial_shape
        return resampled.view(new_shape)
    
    def transform_affine(self, affine: np.ndarray) -> np.ndarray:
        new_affine = affine.copy()
        new_affine[:3, :3] = affine[:3, :3] @ np.diag(1 / self.resample_factor)
        return new_affine
