"""Functions used by the application."""


def stream_selector(tokens):
    result = []
    tokens = tokens.split(",")
    for token in tokens:
        temp = token.split("-")
        if len(temp) == 1:
            result.append(int(token)-1)
        else:
            for i in range(int(temp[0]), int(temp[1])+1):
                result.append(i-1)
    return result
