from config import CURRENT_EXP_DIR, config, get_logger, log_config

logger = get_logger(exp_dir=CURRENT_EXP_DIR)
log_config(logger)

import numpy as np
import sherpa
import sherpa.algorithms.bayesian_optimization as bayesian_optimization
from cade.metrics.comparative import intersection_nn, lncs2, moving_lncs2
from gensim.models.word2vec import Word2Vec
from scipy.spatial.distance import cosine
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def fitness(threshold: float, topn: int, t: float):

    similarity = 3
    # Task 1 - Binary Classification
    predictions = []
    for word in binary_truth[:, 0]:
        if similarity == 0:  # cosine similarity
            prediction = (
                0 if 1 - cosine(model1[word], model2[word]) >= threshold else 1
            )
        elif similarity == 1:  # lncs2 similarity
            prediction = (
                0 if lncs2(word, model1, model2, topn) >= threshold else 1
            )
        elif similarity == 2:  # intersection of nearest neighbours
            prediction = (
                1
                if intersection_nn(word, model1, model2, topn) >= threshold
                else 0
            )
        elif similarity == 3:
            prediction = (
                0
                if moving_lncs2(word, model1, model2, topn, t) >= threshold
                else 1
            )
        predictions.append(prediction)

    result_accu = accuracy_score(
        binary_truth[:, 1].astype(int), np.array(predictions),
    )
    result_f1 = f1_score(
        binary_truth[:, 1].astype(int), np.array(predictions),
    )
    logger.info("F1: " + str(result_f1))
    logger.info(
        "PRE: "
        + str(
            precision_score(
                binary_truth[:, 1].astype(int), np.array(predictions),
            )
        )
    )
    logger.info(
        "REC: "
        + str(
            recall_score(
                binary_truth[:, 1].astype(int), np.array(predictions),
            )
        )
    )
    logger.info("ACC: " + str(result_accu))
    return result_accu


if __name__ == "__main__":

    logger.info("AUTOML ON ACCURACY")
    logger.info("MOVING LNCS2")
    config["LANG"].remove("latin")
    config["LANG"].remove("english")

    port = 1

    for lang in config["LANG"]:
        i = 1
        parameters = [
            sherpa.Continuous("threshold", [0.5, 0.85]),
            sherpa.Discrete("topn", [10, 50]),
            sherpa.Continuous("t", [0.0, 1.0]),
        ]
        algorithm = bayesian_optimization.GPyOpt(max_num_trials=40)
        study = sherpa.Study(
            parameters=parameters,
            algorithm=algorithm,
            lower_is_better=False,
            dashboard_port=(9999 - port + 1),
        )
        model1 = Word2Vec.load(
            CURRENT_EXP_DIR.split("_")[0]
            + "_0"
            + "/model/"
            + lang
            + "/corpus1.model"
        )
        model2 = Word2Vec.load(
            CURRENT_EXP_DIR.split("_")[0]
            + "_0"
            + "/model/"
            + lang
            + "/corpus2.model"
        )
        # Load binary truths
        binary_truth = np.loadtxt(
            "./data/"
            + lang
            + "/semeval2020_ulscd_"
            + lang[:3]
            + "/truth/binary.txt",
            dtype=str,
            delimiter="\t",
        )
        for trial in study:
            thr = trial.parameters["threshold"]
            topn = int(trial.parameters["topn"])
            t = trial.parameters["t"]
            logger.info("Best paramter at iteration " + str(i))
            logger.info(
                "thr: " + str(thr) + ", topn: " + str(topn) + ", t: " + str(t)
            )
            accuracy = fitness(thr, topn, t)
            study.add_observation(trial=trial, objective=accuracy, iteration=i)
            if study.should_trial_stop(trial):
                break
            study.finalize(trial=trial)
            i += 1
        port += 1
