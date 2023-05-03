import tarfile


with tarfile.open('data/raw/clntm/scholar_data.tar.gz', 'r:gz') as tar:
    tar.extractall(path='data/raw/clntm/scholar_data')
