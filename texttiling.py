import xml.etree.ElementTree as ET
import os
from nltk.tokenize import TextTilingTokenizer
from pyspark import SparkContext, SparkConf
import re
import logging
import argparse


def clean(doc_path):
    xml_root = ET.parse(doc_path)
    title_node = xml_root.find("./body/body.head/hedline/hl1")
    title = "" if title_node is None else title_node.text
    content = title

    abs_nodes = xml_root.find("./body/body.head/abstract//*")
    if abs_nodes:
        abstract = "\n\n".join([node.text for node in abs_nodes])
    else:
        abstract = ""
    if abstract:
        content += "\n\n" + abstract

    text_nodes = xml_root.findall("./body/body.content/block[@class='full_text']//*")
    if text_nodes:
        text = "\n\n".join([node.text for node in text_nodes])
    else:
        text = ""

    if text:
        content += "\n\n" + text
    return content


def tiling(corpus_direc, doc_direc="docs", segment_direc="segment", xml_file_path='data/truth_data_nyt_2017_v2.3.xml'):
    os.makedirs(doc_direc, exist_ok=True)
    os.makedirs(segment_direc, exist_ok=True)

    xml_root = ET.parse(xml_file_path)
    docs = xml_root.findall(".//docno")
    doc_set = list(set([doc.text for doc in docs]))

    conf = SparkConf().setAppName("text_tiling")
    sc = SparkContext(conf=conf)

    files = sc.parallelize(doc_set, numSlices=24)

    tokenizer = TextTilingTokenizer(k=6)

    def texttiling_doc(doc_id):

        content = clean(os.path.join(corpus_direc, doc_id + ".xml"))
        with open(os.path.join(doc_direc, doc_id), "w") as output:
            output.write(content)
        try:
            segments = tokenizer.tokenize(content)
            output = '\n\n'.join([re.sub(r'\n+', '\t', segment) for segment in segments])
        except ValueError as e:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.DEBUG)
            logger.addHandler(logging.FileHandler("texttiling.log"))
            logger.warning(doc_id + "\t" + repr(e))

            output = re.sub(r'\n+', '\t', content)

        with open(os.path.join(segment_direc, doc_id), "w") as f:
            print(output, file=f)

    files.map(texttiling_doc).collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_direc", type=str, help="directory to the corpus")
    args = parser.parse_args()
    tiling(args.corpus_direc, doc_direc="docs/", segment_direc="segment/",
           xml_file_path='data/truth_data_nyt_2017_v2.3.xml')
    # tiling("/data/nyt/nyt_corpus_flatten/", "docs/", "segment/", 'data/truth_data_nyt_2017_v2.3.xml')
