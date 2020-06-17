import numpy
from cade.metrics.comparative import lncs2
from gensim.models.word2vec import Word2Vec
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from sklearn.metrics import classification_report

from config import CURRENT_EXP_DIR, config, get_logger, log_config

if __name__ == "__main__":
    # CURRENT_EXP_DIR = CURRENT_EXP_DIR.split("_")[0] + "_0"
    logger = get_logger(exp_dir=CURRENT_EXP_DIR)
    log_config(logger)
    logger.info("TEST WITH HAMILTON's LNCS2")
    for lang in config["LANG"]:
        # Load models
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
        binary_truth = numpy.loadtxt(
            "./data/"
            + lang
            + "/semeval2020_ulscd_"
            + lang[:3]
            + "/truth/binary.txt",
            dtype=str,
            delimiter="\t",
        )
        # Task 1 - Binary Classification
        predictions = []
        for word in binary_truth[:, 0]:
            prediction = (
                0
                if lncs2(word, model1, model2, 10) >= config["THRESHOLD"]
                else 1
            )
            predictions.append(prediction)
        logger.info("CLassification score for " + lang)
        logger.info(
            "\n"
            + classification_report(
                binary_truth[:, 1].astype(float),
                numpy.array(predictions),
                target_names=["class 0 (stable)", "class 1 (change)"],
            )
        )
        # Load scores truths
        score_truth = numpy.loadtxt(
            "./data/"
            + lang
            + "/semeval2020_ulscd_"
            + lang[:3]
            + "/truth/graded.txt",
            dtype=str,
            delimiter="\t",
        )
        # Task 2 - Semantic Shift Score
        scores = []
        for word in score_truth[:, 0]:
            score = lncs2(word, model1, model2, 10)
            scores.append(score)
        rho, _ = spearmanr(scores, score_truth[:, 1], nan_policy="raise")
        logger.info("CLassification score for " + lang)
        logger.info("Spearman score for " + lang + ": " + str(rho))
