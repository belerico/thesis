import datetime
import logging
import os

from cade.cade import CADE
from gensim.models.word2vec import Word2Vec

from preprocess import create_compass

config = {
    "LANG": ["latin", "english", "swedish", "german"],
    "EMBEDDING_SIZE": 100,
    "SITER": 10,
    "DITER": 10,
    "PREPROCESS": False,
}


if __name__ == "__main__":
    if not os.path.exists("./experiments"):
        os.makedirs("./experiments")

    new_exp_num = 0
    for exp in os.listdir("./experiments"):
        exp_num = int(exp.split("_")[1])
        if exp_num > exp_num:
            new_exp_num = exp_num
    CURRENT_EXP_DIR = "./experiments/experiment_" + str(new_exp_num)
    if not os.path.exists(CURRENT_EXP_DIR):
        os.makedirs(CURRENT_EXP_DIR)

    logger = logging.getLogger("thesis")
    fh = logging.FileHandler(
        CURRENT_EXP_DIR + "/experiment_" + str(new_exp_num) + ".log"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s|%(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(fh)

    logger.info(
        "Experiment "
        + str(new_exp_num)
        + ": "
        + datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    )
    logger.info("Current config:")
    for k, v in config.items():
        logger.info(k + ": " + str(v))

    for lang in config["LANG"]:
        # Create compass
        logger.info("Creating compass for " + lang.upper())
        if not os.path.exists(CURRENT_EXP_DIR + "/model/" + lang):
            os.makedirs(CURRENT_EXP_DIR + "/model/" + lang)
        create_compass(
            [
                "./data/"
                + lang
                + "/semeval2020_ulscd_"
                + lang[:3]
                + "/corpus1/lemma/corpus1.txt",
                "./data/"
                + lang
                + "/semeval2020_ulscd_"
                + lang[:3]
                + "/corpus2/lemma/corpus2.txt",
            ],
            save_path=CURRENT_EXP_DIR
            + "/model/"
            + lang
            + "/compass_"
            + lang[:3]
            + ".txt",
        )

        aligner = CADE(
            size=config["EMBEDDING_SIZE"],
            opath=CURRENT_EXP_DIR + "/model/" + lang,
            siter=config["SITER"],
            diter=config["DITER"],
        )

        # train the compass: the text should be the concatenation of the text from the slices
        logger.info("Training compass model")
        aligner.train_compass(
            CURRENT_EXP_DIR
            + "/model/"
            + lang
            + "/compass_"
            + lang[:3]
            + ".txt",
            overwrite=True,
        )

        # now you can train slices and they will be already aligned
        # these are gensim word2vec objects
        logger.info("Training slice1 model")
        slice_one = aligner.train_slice(
            "./data/"
            + lang
            + "/semeval2020_ulscd_"
            + lang[:3]
            + "/corpus1/lemma/corpus1.txt",
            save=True,
        )
        logger.info("Training slice2 model")
        slice_two = aligner.train_slice(
            "./data/"
            + lang
            + "/semeval2020_ulscd_"
            + lang[:3]
            + "/corpus2/lemma/corpus2.txt",
            save=True,
        )


# Load model
# model1 = Word2Vec.load("model/arxiv_14.model")
# model2 = Word2Vec.load("model/arxiv_9.model")
