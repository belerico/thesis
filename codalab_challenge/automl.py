import functools
import pprint

import numpy as np
from cade.metrics.comparative import lncs2
from gensim.models.word2vec import Word2Vec
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential
from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from scipy.spatial.distance import cosine
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from config import CURRENT_EXP_DIR, config, get_logger, log_config


def myFirstRun(self, init_rand_configs=None, n_eval=3):
    """
    Performs initial evaluations before fitting GP.

    Parameters
    ----------
    init_rand_configs: list
        Initial random configurations
    n_eval: int
        Number of initial evaluations to perform. Default is 3.

    """
    if init_rand_configs is None:
        self.X = np.empty((n_eval, len(self.parameter_key)))
        self.y = np.empty((n_eval,))
        for i in range(n_eval):
            s_param = self._sampleParam()
            s_param_val = list(s_param.values())
            self.X[i] = s_param_val
            self.y[i] = self.f(**s_param)
    else:
        self.X = np.empty((len(init_rand_configs), len(init_rand_configs[0])))
        self.y = np.empty((len(init_rand_configs),))
        self.init_evals = len(self.y)
        for i in range(len(init_rand_configs)):
            self.X[i] = list(init_rand_configs[i].values())
            self.y[i] = self.f(**init_rand_configs[i])
    self.GP.fit(self.X, self.y)
    self.tau = np.max(self.y)
    if init_rand_configs is None:
        self.history.append([init_rand_configs[np.argmax(self.y)], self.tau])
    else:
        idx_max_param = np.argmax(self.y)
        self.history.append(
            [
                {
                    key: self.X[idx_max_param, idx]
                    for idx, key in enumerate(self.parameter_key)
                },
                self.GP.y[-1],
                self.tau,
            ]
        )


def myUpdateGP(self):
    """
    Updates the internal model with the next acquired point and its evaluation.
    """
    kw = {
        param: int(round(self.best[i]))
        if self.parameter_type[i] == "int"
        else float(self.best[i])
        for i, param in enumerate(self.parameter_key)
    }
    print(kw)
    f_new = self.f(**kw)
    self.GP.update(np.atleast_2d(self.best), np.atleast_1d(f_new))
    self.tau = np.max(self.GP.y)
    self.history.append([kw, self.GP.y[-1], self.tau])


def get_fitness_for_automl(model1, model2, binary_truth, logger):
    def fitness(threshold: float, similarity: str, topn: int):

        # Task 1 - Binary Classification
        predictions = []
        for word in binary_truth[:, 0]:
            if similarity == 0:  # cosine similarity
                prediction = (
                    0
                    if 1 - cosine(model1[word], model2[word]) >= threshold
                    else 1
                )
            elif similarity == 1:  # lncs2 similarity
                prediction = (
                    0 if lncs2(word, model1, model2, topn) >= threshold else 1
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

    return fitness


if __name__ == "__main__":

    logger = get_logger(exp_dir=CURRENT_EXP_DIR)
    log_config(logger)
    logger.info("AUTOML ON ACCURACY")
    furtherEvaluations = 30
    param = {
        "threshold": ("cont", [5e-1, 8e-1]),
        "topn": ("int", [1, 30]),
        "similarity": ("int", [0, 1]),
    }
    init_rand_configs = [{"threshold": 0.7, "topn": 5, "similarity": 1,}]

    for lang in config["LANG"]:
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
        # Creating a GP surrogate model with a Squared Exponantial
        # covariance function, aka kernel
        sexp = squaredExponential()
        sur_model = GaussianProcess(sexp)
        fitness = get_fitness_for_automl(model1, model2, binary_truth, logger)
        # setting the acquisition function
        acq = Acquisition(mode="ExpectedImprovement")

        # creating an object Bayesian Optimization
        bo = GPGO(sur_model, acq, fitness, param, n_jobs=4)
        bo._firstRun = functools.partial(myFirstRun, bo)
        bo.updateGP = functools.partial(myUpdateGP, bo)
        bo._firstRun(init_rand_configs=init_rand_configs)
        bo.logger._printInit(bo)

        bo.run(furtherEvaluations, resume=True)
        best = bo.getResult()
        logger.info(
            "BEST PARAMETERS: "
            + ", ".join([k + ": " + str(v) for k, v in best[0].items()])
            + ", ACCU: "
            + str(best[1])
        )
        logger.info("OPTIMIZATION HISTORY")
        logger.info(pprint.pformat(bo.history))
