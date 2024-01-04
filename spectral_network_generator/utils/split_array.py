def split_array(arr, row_size):
    """
    Parameters
    ----------
    arr : numpy.ndarray
    row_size : int

    Returns
    -------
    tuple(numpy.ndarray, int, int)
        Divided array with 'row_size' rows,  start and end indexes.
    """
    if arr.shape[0] <= row_size:
        yield arr, 0, arr.shape[0]

    else:
        chunk_num = arr.shape[0] // row_size
        for i in range(chunk_num):
            start_index = i * row_size
            end_index = (i + 1) * row_size
            split_arr = arr[start_index:end_index]
            yield split_arr, start_index, end_index

        remainder = arr.shape[0] % row_size
        if remainder:
            start_index = arr.shape[0] - remainder
            split_arr = arr[start_index:]
            yield split_arr, start_index, arr.shape[0]
