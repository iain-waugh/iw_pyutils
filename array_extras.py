#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extend the standard Numpy array manipulation routines,
particularly for 2D array operations.

Copyright (C) 2021 - Iain Waugh, iwaugh@gmail.com
"""

from __future__ import print_function, division
import numpy as np

__version__ = "1.0.0"

def shift_2d(arr, shifts, stride=0):
    """
    Shift a 2D array by rows/columns and either pad the new data with zeros or
    copy existing data from the trailing edge of the shift.

    Parameters
    ----------
    arr : 2D numpy array
        The input arrray

    shift_row
        How many rows to shift up/down

    shift_col
        How many columns to shift left/right

    stride
        How many of the previous rows/columns to repeat

        0 = fill with zeros (default)

        1 = copy the last row/column

        2 = copy the last 2 rows/columns (good for working with Bayer paterns)

    Returns
    -------
    An array that is shifted by 'shift_row' and 'shift_col', with new data determined by 'stride'
    """
    (shift_row, shift_col) = shifts
    row_arr = np.zeros_like(arr)
    if stride > 0:
        quot = abs(shift_row) // stride
        rem = shift_row % stride
        if rem != 0:
            quot += 1

    if shift_row > 0:
        if stride > 0:
            stride_arr = np.tile(arr[:stride, :], (quot, 1))[-shift_row:, :]
            row_arr = np.vstack((stride_arr, arr[:-shift_row, :]))
        else:
            row_arr[shift_row:, :] = arr[:-shift_row, :]

    elif shift_row < 0:
        if stride > 0:
            stride_arr = np.tile(arr[-stride:, :], (quot, 1))[:-shift_row, :]
            row_arr = np.vstack((arr[-shift_row:, :], stride_arr))
        else:
            row_arr[:shift_row, :] = arr[-shift_row:, :]
    else:
        row_arr[:, :] = arr[:, :]

    # Shift the Columns
    if stride > 0:
        quot = abs(shift_col) // stride
        rem = shift_col % stride
        if rem != 0:
            quot += 1
    new_arr = np.zeros_like(arr)
    if shift_col > 0:
        if stride > 0:
            stride_arr = np.tile(row_arr[:, :stride], (1, quot))[:, -shift_col:]
            new_arr = np.hstack((stride_arr, row_arr[:, :-shift_col]))
        else:
            new_arr[:, shift_col:] = row_arr[:, :-shift_col]

    elif shift_col < 0:
        if stride > 0:
            stride_arr = np.tile(row_arr[:, -stride:], (1, quot))[:, :-shift_col]
            new_arr = np.hstack((row_arr[:, -shift_col:], stride_arr))
        else:
            new_arr[:, :shift_col] = row_arr[:, -shift_col:]
    else:
        new_arr[:, :] = row_arr[:, :]

    return new_arr


def add_border_2d(arr, border_width, stride=0):
    """
    Expand the border of a Numpy array by replicating edge values

    Parameters
    ----------
    arr : Numpy ndarray
        The ndarray that you want to enlarge.

    border_width : integer
        How many rows and columns do you want to add to the edge of the ndarray?

    stride : integer
        How many of the previous rows/columns to repeat

        0 = fill with zeros (default)

        1 = copy the last row/column

        2 = copy the last 2 rows/columns (good for working with Bayer paterns)

    Returns
    -------
    A version of `arr` with borders.
    """
    new_x = arr.shape[0] + border_width * 2
    new_y = arr.shape[1] + border_width * 2
    new_dtype = arr.dtype
    new_arr = np.zeros((new_x, new_y), dtype=new_dtype)

    new_arr[border_width:-border_width, border_width:-border_width] = arr
    if stride > 0:
        # We want to copy data from existing rows/columns to into the new border
        quot = border_width // stride
        rem = border_width % stride
        if rem != 0:
            quot += 1

        # Fill the top
        arr_t = new_arr[border_width : border_width + stride, :]
        new_arr[:border_width,] = np.tile(
            arr_t, (quot, 1)
        )[:border_width, :]
        # Fill the botton
        arr_b = new_arr[-(border_width + stride) : -border_width, :]
        new_arr[-border_width:,] = np.tile(
            arr_b, (quot, 1)
        )[:border_width, :]
        # Fill left
        arr_l = new_arr[:, border_width : border_width + stride]
        new_arr[:, :border_width] = np.tile(arr_l, (1, quot))[:, :border_width]
        # Fill the right
        arr_b = new_arr[:, -(border_width + stride) : -border_width]
        new_arr[:, -border_width:] = np.tile(arr_b, (1, quot))[:, :border_width]

    return new_arr


if __name__ == "__main__":
    aRow, aCol = (6, 6)  #  Number of rows and columns
    arr = np.arange(1, aRow * aCol + 1).astype("uint16").reshape(aRow, aCol)

    print("Original array\n", arr)

    n1 = shift_2d(arr, (1, 0), 1)
    s1 = shift_2d(arr, (-1, 0), 1)
    w1 = shift_2d(arr, (0, 1), 1)
    e1 = shift_2d(arr, (0, -1), 1)

    print("North 1\n", n1)
    print("South 1\n", s1)
    print("East 1\n", e1)
    print("West 1\n", w1)

    n2 = shift_2d(arr, (2, 0), 2)
    s2 = shift_2d(arr, (-2, 0), 2)
    w2 = shift_2d(arr, (0, 2), 2)
    e2 = shift_2d(arr, (0, -2), 2)

    print("North 2\n", n2)
    print("South 2\n", s2)
    print("East 2\n", e2)
    print("West 2\n", w2)

    print(
        "Adding a border of 2 rows/columns, copying data from the outermost row/column"
    )
    print(add_border_2d(arr, 2, 1))
    print(
        "Adding a border of 3 rows/columns, copying data from the 3x outermost rows/columns"
    )
    print(add_border_2d(arr, 3, 3))
