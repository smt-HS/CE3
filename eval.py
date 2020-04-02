import subprocess
import shlex
import argparse
import os
import sys
from collections import defaultdict
import json
import statistics
from xml.etree import ElementTree


def generate_run_file(output_file):
    os.makedirs("runs/", exist_ok=True)
    output = open(os.path.join("runs", output_file), "w")
    for file in os.listdir("ds_log"):
        if not (file.startswith("DS-") or file.startswith("RL")):
            continue
        parts = file.split(".")[0].split("-")
        topic, env_rank = "{}-{}".format(parts[2], parts[3]), int(parts[4])
        if env_rank != 0:
            continue
        cmd = "grep ^{} ds_log/{}".format(topic, file)
        result = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        lines = result.stdout.decode("utf-8").splitlines()

        max_iter = -1
        idx, total_len = 0, len(lines)

        while idx < total_len:
            niter = int(lines[idx].split("\t")[1])
            if niter < max_iter:
                break
            else:
                max_iter = niter
                idx += 5  # 5 lines per iteration

        begin, end = None, None
        idx = total_len - 1
        while idx >= 0:
            niter = int(lines[idx].split("\t")[1])
            if niter == max_iter:
                end = idx + 1
                begin = idx + 1 - (max_iter + 1) * 5
                break
            else:
                idx -= 5
        assert begin < end

        output.write("\n".join(lines[begin:end]) + "\n")

    output.close()


def metrics(run_file, cutoff, detail):
    metric_names = []
    scores = []
    output = ""
    for metric in ["cubetest", "sDCG", "expected_utility"]:
        cmd = "python trec-dd-jig/scorer/{}.py --topics=trec-dd-jig/topics/truth_data_nyt_2017_v2.3.xml " \
              "--params=trec-dd-jig/topics/params " \
              "--runfile=runs/{} " \
              "--cutoff={}".format(metric, run_file, cutoff)

        result = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        lines = result.stdout.decode("utf-8").splitlines()
        if not detail:
            parts = lines[1].split("\t")
            metric_names += [parts[1], parts[-1]]  # no act
            # print(parts)
            parts = lines[-1].split("\t")
            scores += [parts[1], parts[-1]]  # no act
        else:
            output += "\n".join(lines[1:]) + "\n"

    if not detail:
        # print(metric_names)
        assert len(scores) == 6
        print("\t".join(metric_names))
        print("\t".join(scores))
    else:
        print(output)

    prec, recall = prec_recall(run_file, cutoff)
    aspect_ratio = aspect(run_file, cutoff)
    print("Precision\tRecall\tAspect\n{}\t{}\t{}".format(prec, recall, aspect_ratio))
    # print("Precision: {}\t Recall: {}\t Aspect: {}".format(prec, recall, aspect_ratio))


def prec_recall(run_file, cutoff):
    topic_doc = defaultdict(set)
    for line in open(os.path.join("runs", run_file)):
        parts = line.split("\t")
        topic_id, iter_cnt, doc = parts[0], int(parts[1]), parts[2]
        if iter_cnt < cutoff:
            topic_doc[topic_id].add(doc)
    prec_list, recall_list = [], []
    for topic_id in topic_doc:
        doc_list = json.load(open("data/small_corpus.json"))[topic_id]
        rel_docs = set(doc_list[:len(doc_list) // 2])

        hit = rel_docs.intersection(topic_doc[topic_id])
        # print(len(topic_doc[topic_id]))
        prec = len(hit) / len(topic_doc[topic_id])
        recall = len(hit) / len(rel_docs)
        prec_list.append(prec)
        recall_list.append(recall)
    # print(prec_list, recall_list)

    return statistics.mean(prec_list), statistics.mean(recall_list)


def aspect(run_file, cutoff):
    topic_doc = defaultdict(set)
    for line in open(os.path.join("runs", run_file)):
        parts = line.split("\t")
        topic_id, iter_cnt, doc = parts[0], int(parts[1]), parts[2]
        if iter_cnt < cutoff:
            topic_doc[topic_id].add(doc)

    ratio_list = []
    for topic_id in topic_doc:
        doc2subtopic = defaultdict(set)
        subtopic_set = set()
        topic_data = ElementTree.parse("data/truth_data_nyt_2017_v2.3.xml").find(
            "./*/topic[@id=\"{}\"]".format(topic_id))
        for subtopic_data in topic_data.findall("./subtopic"):
            subtopic_id = subtopic_data.attrib["id"]
            subtopic_set.add(subtopic_id)
            for doc_data in subtopic_data.findall(".//docno"):
                doc = doc_data.text
                doc2subtopic[doc].add(subtopic_id)

        hit_subtopic = set(subtopic for doc in topic_doc[topic_id] for subtopic in doc2subtopic[doc])
        ratio = len(hit_subtopic) / len(subtopic_set)

        ratio_list.append(ratio)
    return statistics.mean(ratio_list)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("func")
    parser.add_argument("--begin-iter", type=int, default=None)
    parser.add_argument("--end-iter", type=int, default=None)
    parser.add_argument("--run", type=str, default="")
    parser.add_argument("--cutoff", type=int, default=5)
    parser.add_argument("--detail", action="store_true")

    args = parser.parse_args()
    if args.func == "generate":
        assert args.run != ""
        generate_run_file(args.run)
    elif args.func == "metrics":
        metrics(args.run, args.cutoff, args.detail)
    else:
        raise RuntimeError("Illegal option: {}".format(args.func))


if __name__ == "__main__":
    run()
