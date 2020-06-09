import logging
import os

from cade.cade import CADE
from gensim.models.word2vec import Word2Vec

from preprocess import create_compass

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


LANG = ["english", "latin", "swedish", "german"]
EMBEDDING_SIZE = 100
SITER = 5
DITER = 5

if __name__ == "__main__":
    for lang in LANG:
        # Create compass
        logging.info({"Creating compass for " + lang.upper()})
        if not os.path.exists("./model/" + lang):
            os.makedirs("./model/" + lang)
        create_compass(
            [
                "./data/"
                + lang
                + "/semeval2020_ulscd_"
                + lang[:3]
                + "/corpus1/lemma/ccoha1.txt",
                "./data/"
                + lang
                + "/semeval2020_ulscd_"
                + lang[:3]
                + "/corpus2/lemma/ccoha2.txt",
            ],
            save_path="./model/" + lang + "/compass_" + lang[:3] + ".txt",
        )

        aligner = CADE(
            size=EMBEDDING_SIZE,
            opath="./model/" + lang,
            siter=SITER,
            diter=DITER,
        )

        # train the compass: the text should be the concatenation of the text from the slices
        aligner.train_compass(
            "./model/" + lang + "/compass_" + lang[:3] + ".txt", overwrite=True
        )

        # now you can train slices and they will be already aligned
        # these are gensim word2vec objects
        slice_one = aligner.train_slice(
            "./data/"
            + lang
            + "/semeval2020_ulscd_"
            + lang[:3]
            + "/corpus1/lemma/ccoha1.txt",
            save=True,
        )
        slice_two = aligner.train_slice(
            "./data/"
            + lang
            + "/semeval2020_ulscd_"
            + lang[:3]
            + "/corpus2/lemma/ccoha2.txt",
            save=True,
        )


# Load model
# model1 = Word2Vec.load("model/arxiv_14.model")
# model2 = Word2Vec.load("model/arxiv_9.model")
