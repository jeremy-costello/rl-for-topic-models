import os
from bs4 import BeautifulSoup


def main():
    root = 'data/raw/nyt/nyt_corpus/data'
    output_file = 'data/raw/nyt/nyt_output.txt'

    with open(output_file, 'w') as f:
        f.write('')

    with open(output_file, 'a') as f:
        for path, subdirs, files in os.walk(root):
            for name in files:
                if name.split('.')[-1] == 'xml':
                    full_path = os.path.join(path, name)
                    article_text = parse_xml(full_path)
                    if article_text:
                        f.write(f'{article_text}\n')
                    else:
                        print(full_path)


def parse_xml(filename):
    with open(filename, 'r') as f:
        file = f.read()

    soup = BeautifulSoup(file, 'xml')

    headline = soup.find('hedline')
    if headline is None:
        headline_text = ''
    else:
        headline_hl1 = headline.hl1
        if headline_hl1 is None:
            headline_text = ''
        else:
            headline_text = headline_hl1.text.strip()

    body_text_block = soup.find('block', {'class': 'full_text'})
    if body_text_block is None:
        body_text = ''
    else:
        ps = body_text_block.find_all('p')
        body_text_list = [p.text.strip() for p in ps]
        body_text = ' '.join(body_text_list)

    full_text = headline_text + ' ' + body_text
    return full_text.strip()


if __name__ == '__main__':
    main()
