import ast


def load_n_grams(file_path):
    with open(file_path) as fr:
        words = fr.read()
        words = ast.literal_eval(words)
    return words
