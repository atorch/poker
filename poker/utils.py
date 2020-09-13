def argmax(values):

    """
    Return the _list_ of indexes that maximize a list of non-negative values
    For example, argmax([0, 55, 99, 99]) is [2, 3]
    """

    max_value = -1
    argmax_indexes = []

    for index, value in enumerate(values):
        if value == max_value:
            argmax_indexes.append(index)
        elif value > max_value:
            argmax_indexes = [index]
            max_value = value

    return argmax_indexes
