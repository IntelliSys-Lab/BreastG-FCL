import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scipy import linalg

logger = logging.getLogger("GFedCL")


class SimplifiedFID:
    """
    Simplified FID calculator using a ResNet-18 feature extractor.
    """

    def __init__(self, device="cuda"):
        self.device = device
        try:
            model = models.resnet18(pretrained=True)
            model.fc = nn.Identity()
            model = model.to(device)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            self.model = model
            logger.info("SimplifiedFID initialized with ResNet-18")
        except Exception as exc:
            logger.error(f"Error initializing ResNet-18 for FID: {exc}")
            self.model = None

    @torch.no_grad()
    def extract_features(self, dataloader):
        if self.model is None:
            return None

        features = []
        try:
            for data in dataloader:
                if isinstance(data, (tuple, list)):
                    inputs = data[0].to(self.device)
                else:
                    inputs = data.to(self.device)

                if inputs.shape[1] != 3:
                    logger.warning(
                        f"Expected 3 channels but got {inputs.shape[1]}. Skipping batch."
                    )
                    continue

                if inputs.shape[2] < 32 or inputs.shape[3] < 32:
                    inputs = F.interpolate(
                        inputs, size=(32, 32), mode="bilinear", align_corners=False
                    )

                batch_features = self.model(inputs)
                features.append(batch_features.cpu())
        except Exception as exc:
            logger.error(f"Error extracting features: {exc}")
            return None

        if not features:
            return None

        try:
            return torch.cat(features, dim=0)
        except Exception as exc:
            logger.error(f"Error concatenating features: {exc}")
            return None

    def calculate_fid(self, real_dataloader, fake_dataloader):
        real_features = self.extract_features(real_dataloader)
        fake_features = self.extract_features(fake_dataloader)

        if real_features is None or fake_features is None:
            logger.warning("Failed to extract features for FID calculation")
            return None

        if len(real_features) < 2 or len(fake_features) < 2:
            logger.warning(
                "Not enough samples for FID calculation: "
                f"{len(real_features)} real, {len(fake_features)} fake"
            )
            return None

        try:
            return calculate_fid(real_features, fake_features)
        except Exception as exc:
            logger.error(f"Error calculating FID: {exc}")
            return None


def calculate_fid(real_features, fake_features):
    """
    Calculate Frechet Inception Distance between two feature distributions.
    """
    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)

    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(fake_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(fake_features, rowvar=False)

    diff = mu1 - mu2

    try:
        covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)[0]
    except Exception:
        covmean = None

    if covmean is None or not np.isfinite(covmean).all():
        eps = 1e-6
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)[0]

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(fid)
