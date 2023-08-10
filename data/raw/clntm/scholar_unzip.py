from zipfile import ZipFile


with ZipFile('data/raw/clntm/data.zip', 'r') as zip:
    zip.extractall('data/raw/clntm/scholar_data/data')
