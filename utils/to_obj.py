from typing import List, Tuple

import cv2
import numpy as np
import trimesh

def make_point_cloud(array: np.ndarray, spacings: List[float], color=None) -> np.array:

    array = array.astype(np.uint8)
    depth = array.shape[-1]
    spacings = np.array(spacings)

    vertices = []
    for i in range(depth):
        contours = cv2.findContours(
            array[..., i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        for contour in contours:
            contour = contour[:, 0]
            for point in contour:
                vertex = [point[0], point[1], i]
                if color is not None:
                    vertex += list(color)
                vertex = np.array(vertex).astype(float)
                vertex[:3] *= spacings
                vertices.append(vertex)

    if len(vertices) > 0:
        vertices = np.stack(vertices)

    return vertices

def array_to_obj(array: np.ndarray, spacings: List[float], color: Tuple[int], filename: str):
    """
    Convert a NumPy array to an object file.

    Parameters
    ----------
    array : np.ndarray
        The NumPy array to be converted, should be of shape (Width, Height, Depth).
    spacings : List[float]
       The spacing between elements in each dimension of the array.
    color: Tuple[int]
       The color of the elements in the array.
    filename : str
        The name of the output object file.

    """
    vertices = make_point_cloud(array, spacings, color)
    mesh = trimesh.Trimesh(vertices, faces=[])
    mesh.export(filename)
