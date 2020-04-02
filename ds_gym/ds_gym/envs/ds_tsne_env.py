import gym
from gym.spaces import *
# import gensim
import os
import numpy as np
import json
from xml.etree import ElementTree
from collections import Counter
import logging
import datetime
import sys
import math


class DStSNEEnv(gym.Env):
    metadata = {'render.modes': ['ansi']}
    segment_num = 20
    tsne_dim = 3

    def __init__(self):
        # placeholders
        self.topic_id = ""
        self.observation_space = Box(low=-10, high=10, shape=(20, DStSNEEnv92.segment_num, DStSNEEnv92.tsne_dim),
                                     dtype=float)
        self.action_space = Box(low=-1, high=1, shape=(DStSNEEnv92.tsne_dim,), dtype=float)
        # self.dct = gensim.corpora.Dictionary()
        # self.state = np.asarray([])  # tensor (#doc, #segment, 1)
        self.board = np.asarray([])  # tensor (#doc,  #segment, 3)
        self.board_backup = np.copy(self.board)
        # self.vocab_size = 0
        self.doc_num = 0
        self.doc_list = []
        # self.cupy_device = 0
        self.doc_rel = Counter()  # doc -> relevance score
        self.submitted = set()
        self.doc2idx = {}

        self.iter_cnt = 0
        self.acc_reward = 0
        self.logger = logging.getLogger(__name__)  # place holder, will be changed later
        self.name = self.__class__.__name__
        self.empty_value = np.random.random((DStSNEEnv92.segment_num, DStSNEEnv92.tsne_dim)) * 10
        # self.play_log = None
        # self.play = False

    def _set_logger(self, topic_id, env_rank):
        # file handler
        os.makedirs('ds_log/', exist_ok=True)
        fh = logging.FileHandler("ds_log/RL-{}-{}-{}-{}.log".format(
            self.name, topic_id, env_rank, datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
        fh.setLevel(logging.DEBUG)

        # setup logger
        self.logger = logging.getLogger("{}_{}_{}".format(self.name, topic_id, env_rank))
        self.logger.setLevel(logging.DEBUG)
        # console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.CRITICAL)
        # set format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.info("Logger Created")

    def set_topic(self, topic_id, env_rank, corpus_path, truth_path):
        self.topic_id = topic_id
        self._set_logger(topic_id, env_rank)

        self.logger.info("Running on Topic {}, Env Rank: {}".format(self.topic_id, env_rank))
        # self.cupy_device = gpu_id
        self.doc_list = json.load(open("data/small_corpus.json"))[topic_id]
        self.doc_num = len(self.doc_list)

        # set up doc2idx
        for idx, doc in enumerate(self.doc_list):
            self.doc2idx[doc] = idx

        # self.dct = gensim.corpora.Dictionary.load_from_text(os.path.join(corpus_path, topic_id, "dictionary.txt"))
        # self.vocab_size = len(self.dct)
        self.board = np.load(os.path.join(corpus_path, topic_id, "mat_norm_3.npz"))['tsne_norm']
        self.board = self.board.reshape((self.doc_num, DStSNEEnv92.segment_num, DStSNEEnv92.tsne_dim))
        self.board_backup = np.copy(self.board)

        self.observation_space = Box(low=np.min(self.board), high=np.max(self.board), shape=self.board.shape,
                                     dtype=float)
        self.action_space = Box(low=-1, high=1, shape=(DStSNEEnv92.tsne_dim,), dtype=float)

        # initial action

        # action = np.random.random((DStSNEEnv9.tsne_dim,))

        topic_data = ElementTree.parse(truth_path).find("./*/topic[@id=\"{}\"]".format(topic_id))

        for psg in topic_data.findall(".//passage"):
            doc = psg.find("./docno").text
            rel = int(psg.find("./rating").text)
            self.doc_rel[doc] += rel
        self.submitted.clear()
        self.iter_cnt = 0
        self.acc_reward = 0

        # self.play = play
        # if play:
        #     self.play_log = open(os.path.join("ds_log", "{}_{}_play.log".format(self.topic_id, env_rank)))

    def log_submit(self, submit_docs):
        log_str = ""
        for doc, score, rel in submit_docs:
            log_str += "{}\t{}\t{}\t{}\t{}\t\n".format(self.topic_id, self.iter_cnt, doc, score, rel)
        self.logger.info("\n" + log_str)

        # if self.play:
        #     self.play_log.write(log_str)

    def step(self, action):
        self.logger.info("Action: {}".format(action))

        rel = np.sum(np.dot(self.board, action), axis=1)

        doc_score = sorted(zip(self.doc_list, rel), key=lambda x: -x[1])

        submit = []
        reward = 0
        for doc, score in doc_score:
            if doc not in self.submitted:
                rel = self.doc_rel[doc]
            else:
                rel = 0
            # set selected documents to reserved values
            self.board[self.doc2idx[doc]] = self.empty_value
            submit.append((doc, score, rel))
            self.submitted.add(doc)
            reward += rel
            if len(submit) == 5:
                break

        observation = np.copy(self.board)

        self.log_submit(submit)
        self.acc_reward += reward
        self.iter_cnt += 1
        # half of the corpus are relevant
        done = (len(self.submitted) >= self.doc_num // 2)  or (self.iter_cnt >= 30)

        if done:
            self.logger.warning("Done at iteration {}, total reward {}\n".format(self.iter_cnt, self.acc_reward))

        return observation, reward, done, {"submit": submit, "return": self.acc_reward}

    def reset(self):
        self.submitted.clear()

        self.iter_cnt = 0
        self.acc_reward = 0

        self.board = np.copy(self.board_backup)

        observation = np.copy(self.board)  # .get()
        return observation

    def render(self, mode='ansi', close=False):
        return ""

    def seed(self, seed=None):
        # with cp.cuda.Device(self.cupy_device):
        np.random.seed(seed)

    def close(self):
        # if self.play:
        #     self.play_log.close()
        return
