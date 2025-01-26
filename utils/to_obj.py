from typing import List

import cv2
import numpy as np
import trimesh


def array_to_obj(array: np.ndarray, spacings: List[float], filename: str):
    """
    Convert a NumPy array to an object file.

    Parameters
    ----------
    array : np.ndarray
        The NumPy array to be converted, should be of shape (Width, Height, Depth).
    spacings : List[float]
       The spacing between elements in each dimension of the array.
    filename : str
        The name of the output object file.

    """

    array = array.astype(np.uint8)
    depth = array.shape[-1]
    spacings = np.array(spacings)

    vertices = []
    faces = []
    for i in range(depth):
        contours = cv2.findContours(
            array[..., i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        for contour in contours:
            contour = contour[:, 0]
            for point in contour:
                vertex = np.array([point[0], point[1], i])
                vertex *= spacings
                vertices.append(vertex)

    if len(vertices) > 0:
        vertices = np.stack(vertices)

    mesh = trimesh.Trimesh(vertices, faces=faces)
    mesh.export(filename)
