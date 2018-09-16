import os


def clean_html(html):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html)
    text = soup.get_text()
    return text


def clean_html_file(input_path, output_path, should_tokenize=False, tokenizer=None):
    if os.path.exists(output_path):
        raise Exception("Output path existed")
    with open(input_path, 'r') as fr:
        html = fr.read()
        text = clean_html(html)

    lines = text.split('\n')
    with open(output_path, 'w') as fw:
        for line in lines:
            if len(line.strip()) > 0:
                if should_tokenize and (tokenizer is not None):
                    line = " ".join(tokenizer.syllablize(line.strip()))
                    line = tokenizer.get_tokenized(line)
                fw.write(line + "\n")


def clean_files_from_dir(input_dir, output_dir, should_tokenize=False, tokenizer=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_files = os.listdir(input_dir)
    for input_file in input_files:
        input_file_path = os.path.join(input_dir, input_file)
        if input_file.startswith('.') or os.path.isdir(input_file_path):
            continue
        output_file_path = os.path.join(output_dir, input_file)
        clean_html_file(input_file_path, output_file_path, should_tokenize=should_tokenize, tokenizer=tokenizer)


def test_clean_file():
    data_path = '../data/word_embedding/samples/html/html_data.txt'
    output_path = '../data/word_embedding/samples/training/data.txt'
    clean_html_file(data_path, output_path)


def test_clean_files_in_dir():
    input_dir = '../data/word_embedding/real/html'
    output_dir = '../data/word_embedding/real/training'
    from tokenization.crf_tokenizer import CrfTokenizer
    crf_config_root_path = "../tokenization/"
    crf_model_path = "../models/pretrained_tokenizer.crfsuite"
    tokenizer = CrfTokenizer(config_root_path=crf_config_root_path, model_path=crf_model_path)
    clean_files_from_dir(input_dir, output_dir, should_tokenize=True, tokenizer=tokenizer)


if __name__ == '__main__':
    # test_clean_file()
    test_clean_files_in_dir()
