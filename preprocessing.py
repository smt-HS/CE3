import gensim
from threading import Thread
import os
from collections import OrderedDict
import logging
import json
import nltk
from nltk.corpus import stopwords as stopword_list
import numpy as np
import sklearn.manifold
import multiprocessing.dummy
import shutil
import xml.etree.ElementTree as ET
import random

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stemmer = nltk.stem.PorterStemmer()
stopwords = set(stopword_list.words('english'))


def tokenize(string):
    # tokenize, stem, normalize and remove stopwords
    words = tokenizer.tokenize(string.lower())
    words = [w for w in words if w not in stopwords]
    words = [stemmer.stem(word) for word in words]
    return words


def make_small_corpus():
    if os.path.exists("small_corpus/"):
        shutil.rmtree("small_corpus/")
    os.makedirs("small_corpus/", exist_ok=False)
    xml_file_path = 'data/truth_data_nyt_2017_v2.3.xml'
    xml_root = ET.parse(xml_file_path)

    docs = xml_root.findall(".//docno")
    total_doc_set = set([doc.text for doc in docs])

    doc_list = OrderedDict()

    for topic in range(1, 61):
        topic_id = "dd17-{}".format(str(topic))
        os.makedirs("small_corpus/" + topic_id, exist_ok=True)
        topic_data = xml_root.find("./*/topic[@id=\"{}\"]".format(topic_id))
        doc_nodes = topic_data.findall('.//docno')
        rel_docs = set([node.text for node in doc_nodes])

        irrel_pool = total_doc_set - rel_docs

        irrel_docs = random.sample(irrel_pool, len(rel_docs))

        threads = []
        for doc in rel_docs.union(irrel_docs):
            th = Thread(target=shutil.copy, args=["segment/" + doc, "small_corpus/" + topic_id + "/"], daemon=True)
            threads.append(th)
            th.start()

        doc_list[topic_id] = list(rel_docs) + list(irrel_docs)

        for th in threads:
            th.join()

    json.dump(doc_list, open("data/small_corpus.json", "w"), indent=2)


def build_bow(topic_id):
    max_segment = 20
    topic = "dd17-{}".format(topic_id)
    input_direc = "small_corpus/{}".format(topic)
    output_direc = "corpus_bow/{}".format(topic)
    doc_list = json.load(open("data/small_corpus.json"))[topic]
    # first pass, build dictionary
    logging.warning("Processing {}".format(input_direc))
    dct = gensim.corpora.Dictionary()
    content_mem = OrderedDict()
    for file in os.listdir(input_direc):
        content = open(os.path.join(input_direc, file)).read()
        content_mem[file] = content
        dct.add_documents([tokenize(content)])

    os.makedirs(output_direc, exist_ok=True)

    dct.filter_extremes(no_below=2)

    dct.save_as_text(str(os.path.join(output_direc, "dictionary.txt")))

    vocab_size = len(dct)

    # second pass, build BOW representation per segment
    bow_docs = open(os.path.join(output_direc, "docs.txt"), "w")
    doc2seg_mat = {}
    for doc_id, content in content_mem.items():

        segments = content.split("\n\n")
        # doc_str = ""
        seg_list = []
        for segment in segments:
            tokens = tokenize(segment)
            bow = dct.doc2bow(tokens)
            seg_list.append(bow)
            # ids = dct.doc2idx(tokens)
            # doc_str += ' [' + ' '.join([str(idx) for idx in ids]) + ']'

        bow_docs.write("{}:{}\n".format(doc_id, json.dumps(seg_list)))

        seg_mat = gensim.matutils.corpus2dense(seg_list, vocab_size).transpose()  # (#segment, #terms)
        seg_num = seg_mat.shape[0]

        if seg_num > max_segment:
            avg_exceeds = np.average(seg_mat[max_segment - 1:, :], axis=0)
            seg_mat[max_segment - 1] = avg_exceeds
            seg_mat = seg_mat[:max_segment, :]
        elif seg_num < max_segment:
            new_seg_mat = np.pad(seg_mat, ((0, int(max_segment - seg_num)), (0, 0)), mode='constant', constant_values=0)
            seg_mat = new_seg_mat
        doc2seg_mat[doc_id] = seg_mat

    bow_docs.close()

    board = np.asarray([doc2seg_mat[doc] for doc in doc_list])

    # np.save(os.path.join(output_direc, "board"), board)

    np.savez_compressed(os.path.join(output_direc, "board"), board=board)

    norm = board / (np.sum(board, axis=2, keepdims=True) + 1.0)  # smoothing
    np.savez_compressed("corpus_bow/dd17-{}/board.npz".format(topic_id), board=board, norm=norm)

    logging.warning("{} Done!".format(input_direc))


def dim_reduction_norm_single_core(topic_id):
    logging.warning("Topic {} Begins ...".format(topic_id))
    board = np.load("corpus_bow/dd17-{}/board.npz".format(topic_id))['norm']
    doc_num, segment_num, vocab_size = board.shape
    board = board.reshape((doc_num * segment_num, vocab_size))

    # tsne
    tsne = sklearn.manifold.TSNE(perplexity=5, n_components=3).fit_transform(board)

    np.savez_compressed("corpus_bow/dd17-{}/mat_norm_3.npz".format(topic_id), tsne_norm=tsne)
    logging.warning("Topic {} Done!".format(topic_id))


def main():
    make_small_corpus()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)
    threads = []

    for topic_id in range(1, 61):
        th = Thread(target=build_bow, args=[topic_id], daemon=True)
        # dim_reduction(topic_id, topic_id % 4)
        # svd(topic_id, topic_id % 4)
        threads.append(th)
        th.start()

    for th in threads:
        th.join()

    pool = multiprocessing.Pool(processes=4)
    pool.map(dim_reduction_norm_single_core, range(1, 61))


if __name__ == "__main__":
    main()
