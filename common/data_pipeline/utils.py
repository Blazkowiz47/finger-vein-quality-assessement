import numpy as np


def split_image(image: np.ndarray, N: int = 6) -> np.ndarray:
    """
    splits (H,W) image into N * N blocks
    Returns: np.ndarray of shape: ( H // N, W  // N, N*N )
    """
    blocks = []
    for h in range(0, 60, N):
        block_row = []
        for w in range(0, 120, N):
            slice_img = [[image[ih][iw] for iw in range(w, w + N)] for ih in range(h, h + N)]
            block_row.append(slice_img)
        blocks.append(block_row)
    blocks = np.array(blocks)
    x, y, _, _ = blocks.shape
    blocks = blocks.reshape((x, y, N * N))
    return blocks
