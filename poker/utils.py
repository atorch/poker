def argmax(values):

    max_value = -1
    argmax = []

    for index, value in enumerate(values):
        if value == max_value:
            argmax.append(index)
        elif value > max_value:
            argmax = [index]
            max_value = value

    return argmax
