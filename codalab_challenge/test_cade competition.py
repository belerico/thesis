import os
import subprocess
from distutils.dir_util import copy_tree

import numpy
from gensim.models.word2vec import Word2Vec
from scipy.spatial.distance import cosine

from config import CURRENT_EXP_DIR, config

if __name__ == "__main__":
    CURRENT_EXP_DIR = CURRENT_EXP_DIR.split("_")[0] + "_0"
    if not os.path.exists(CURRENT_EXP_DIR + "/res/answer/task1"):
        os.makedirs(CURRENT_EXP_DIR + "/res/answer/task1")
    if not os.path.exists(CURRENT_EXP_DIR + "/res/answer/task2"):
        os.makedirs(CURRENT_EXP_DIR + "/res/answer/task2")
    if not os.path.exists(CURRENT_EXP_DIR + "/ref"):
        copy_tree("./data/ref", CURRENT_EXP_DIR + "/ref")
    # logger = get_logger(exp_dir=CURRENT_EXP_DIR, exp_num=0)
    for lang in config["LANG"]:
        # Load models
        model1 = Word2Vec.load(
            CURRENT_EXP_DIR + "/model/" + lang + "/corpus1.model"
        )
        model2 = Word2Vec.load(
            CURRENT_EXP_DIR + "/model/" + lang + "/corpus2.model"
        )
        # Load binary truths
        binary_truth = numpy.loadtxt(
            "./data/"
            + lang
            + "/semeval2020_ulscd_"
            + lang[:3]
            + "/targets.txt",
            dtype=str,
            delimiter="\t",
        )
        # Task 1 - Binary Classification
        predictions = []
        for word in binary_truth:
            prediction = (
                0 if 1 - cosine(model1[word], model2[word]) >= 0.7 else 1
            )
            predictions.append([word, str(prediction)])
        numpy.savetxt(
            CURRENT_EXP_DIR + "/res/answer/task1/" + lang + ".txt",
            numpy.array(predictions),
            fmt="%s",
            delimiter="\t",
        )
        # Load scores truths
        score_truth = numpy.loadtxt(
            "./data/"
            + lang
            + "/semeval2020_ulscd_"
            + lang[:3]
            + "/targets.txt",
            dtype=str,
            delimiter="\t",
        )
        # Task 2 - Semantic Shift Score
        scores = []
        for word in score_truth:
            score = 1 - cosine(model1[word], model2[word])
            scores.append([word, score])
        numpy.savetxt(
            CURRENT_EXP_DIR + "/res/answer/task2/" + lang + ".txt",
            numpy.array(scores),
            fmt="%s",
            delimiter="\t",
        )
    subprocess.run(["./scoring_program/evaluation.py", CURRENT_EXP_DIR, CURRENT_EXP_DIR])