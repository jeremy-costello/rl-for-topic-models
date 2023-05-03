tar -xvzf nyt_corpus_LDC2008T19.tgz

cd nyt_corpus/data
find . -name '*.tgz' -exec tar -xvzf {} \;

cd ../..
python nyt_xml_to_text.py