
import nltk
import os


def nltk_find_and_download(package_name, path):
    downloaded = False
    try:
        if (nltk.data.find(f'{path}/{package_name}')):
            downloaded = True
    except:
        pass
    try:
        if nltk.data.find(f'{path}/{package_name}.zip'):
            downloaded = True
    except:
        pass

    if not downloaded:
        nltk.download(package_name)


def download_nltk_data():
    nltk_find_and_download('wordnet', 'corpora')
    nltk_find_and_download('punkt', 'tokenizers')
    nltk_find_and_download('averaged_perceptron_tagger', 'taggers')


if int(os.environ.get("RANK", 0)) == 0:
    download_nltk_data()
