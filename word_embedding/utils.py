import os
import urllib.request
__author__ = "Cao Bot"
__copyright__ = "Copyright 2018, DeepAI-Solutions"


def clean_script(html):
    """
    Clean html tags, scripts and css code
    :param html: input html content
    :return: cleaned text
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html)
    for script in soup(["script", "style"]):
        script.extract()
    # get text
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text


def download_html(url_path, output_path, should_clean=True):
    """
    Download html content from url
    :param url_path: input url
    :param output_path: path to output file
    :param should_clean: should clean or not
    :return: cleaned text
    """
    with urllib.request.urlopen(url_path) as response:
        html = response.read()
        if should_clean:
            text = clean_script(html)
        else:
            text = html
    with open(output_path, 'w') as fw:
        fw.write(text)
    return text


def clean_html(html):
    """
    Clean html tags only
    :param html: input html content
    :return: clean text
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html)
    text = soup.get_text()
    return text


def clean_html_file(input_path, output_path, should_tokenize=False, tokenizer=None):
    """
    Clean html tags, script in a file
    :param input_path: path to input file
    :param output_path: path to output file
    :param should_tokenize: should tokenize text or not
    :param tokenizer: if should_tokenize is True, you have to provide tokenizer
    :return: None
    """
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
    """
    Clean html tags scripts for files in a directory
    :param input_dir: input directory
    :param output_dir: output directory
    :param should_tokenize: should tokenize text or not?
    :param tokenizer: tokenizer (required when should_tokenize is True)
    :return: None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_files = os.listdir(input_dir)
    for input_file in input_files:
        input_file_path = os.path.join(input_dir, input_file)
        if input_file.startswith('.') or os.path.isdir(input_file_path):
            continue
        output_file_path = os.path.join(output_dir, input_file)
        clean_html_file(input_file_path, output_file_path, should_tokenize=should_tokenize, tokenizer=tokenizer)


"""Tests"""


def test_download_html():
    url_path = "https://dantri.com.vn/su-kien/anh-huong-bao-so-6-dem-nay-mot-so-tinh-dong-bac-bo-co-gio-giat-manh-20180916151250555.htm"
    output_path = "../data/word_embedding/real/html/html_data.txt"
    download_html(url_path, output_path)


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
    test_download_html()
    # test_clean_file()
    # test_clean_files_in_dir()
