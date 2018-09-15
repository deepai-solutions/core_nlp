import ast
import os


def load_n_grams(file_path):
    with open(file_path) as fr:
        words = fr.read()
        words = ast.literal_eval(words)
    return words


def clean_html(html):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html)
    text = soup.get_text()
    return text


def clean_html_file(input_path, output_path):
    if os.path.exists(output_path):
        raise Exception("Output path existed")
    with open(input_path, 'r') as fr:
        html = fr.read()
        text = clean_html(html)

    lines = text.split('\n')
    with open(output_path, 'w') as fw:
        for line in lines:
            if len(line.strip()) > 0:
                fw.write(line + "\n")


def test_clean_file():
    data_path = '../data/tokenized/samples/html/html_data.txt'
    output_path = '../data/tokenized/samples/training/data.txt'
    clean_html_file(data_path, output_path)


if __name__ == '__main__':
    test_clean_file()
