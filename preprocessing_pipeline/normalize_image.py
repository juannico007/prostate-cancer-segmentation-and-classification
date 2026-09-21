import itk
import numpy as np
from typing import Literal

def normalize_image(image: itk.Image, method: Literal["zscore", "minmax", "none"] = "none") -> itk.Image:
    """Return a normalised copy of ``image`` using the requested strategy.

    Parameters
    ----------
    image : itk.Image
        Input image to normalise. Spatial metadata is preserved in the output.
    method : {"zscore", "minmax", "none"}, default="none"
        Name of the intensity scaling strategy to apply.

    Returns
    -------
    itk.Image
        The normalised image (or the original image when ``method`` is ``none``).
    """
    if method == "none":
        return image

    arr = itk.GetArrayFromImage(image).astype(np.float32)

    if method == "zscore":
        m, s = float(arr.mean()), float(arr.std())
        arr = (arr - m) / s if s > 0 else (arr - m)
    elif method == "minmax":
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) if hi > lo else (arr - lo)
    else:
        raise ValueError(f"Unknown normalization: {method}")

    out = itk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out