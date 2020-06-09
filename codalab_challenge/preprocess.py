import logging
import os

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


def create_compass(
    corpuses: list, save_path="./data/compass/compass.txt", preprocess=False
):
    """Create compass file to train CADE model

    Args:
        corpuses (list): list of corpuses filenames
        save_path (str, optional): where to save the compass file.
            Defaults to "./data/compass/compass.txt".
        preprocess (bool, optional): whether to preprocess every file.
            Defaults to False.
    """
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    with open(save_path, "w+") as f:
        for corpus in corpuses:
            logging.info("Loading file: " + corpus)
            corpus = open(corpus, "r").read()
            logging.info("Writing to compass file: " + save_path)
            f.write(corpus + "\n")
