import datetime
import os

from cade.cade import CADE

from config import CURRENT_EXP_DIR, NEW_EXP_NUM, config, get_logger
from preprocess import create_compass

if __name__ == "__main__":
    logger = get_logger()
    logger.info(
        "Experiment "
        + str(NEW_EXP_NUM)
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

        # train the compass: the text should be the concatenation
        # of the text from the slices
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
