import logging
import os

config = {
    "LANG": ["latin", "english", "swedish", "german"],
    "EMBEDDING_SIZE": 100,
    "SITER": 10,
    "DITER": 10,
    "PREPROCESS": False,
}

if not os.path.exists("./experiments"):
    os.makedirs("./experiments")

NEW_EXP_NUM = 0
for exp in os.listdir("./experiments"):
    exp_num = int(exp.split("_")[1])
    if exp_num > exp_num:
        NEW_EXP_NUM = exp_num
NEW_EXP_NUM += 1
CURRENT_EXP_DIR = "./experiments/experiment_" + str(NEW_EXP_NUM)
if not os.path.exists(CURRENT_EXP_DIR):
    os.makedirs(CURRENT_EXP_DIR)


def get_logger(exp_dir=None, exp_num=None):
    logging.basicConfig(
        filename=(
            (CURRENT_EXP_DIR if exp_dir is None else exp_dir)
            + "/experiment_"
            + (str(NEW_EXP_NUM) if exp_num is None else str(exp_num))
            + ".log"
        ),
        filemode="a",
        level=logging.DEBUG,
        format="%(asctime)s|%(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger()
    # logger.handlers = []
    # fh = logging.FileHandler(
    #     (CURRENT_EXP_DIR if exp_dir is None else exp_dir)
    #     + "/experiment_"
    #     + (str(NEW_EXP_NUM) if exp_num is None else str(exp_num))
    #     + ".log",
    #     "porcodio.txt",
    #     mode="a",
    # )
    # fh.setLevel(logging.DEBUG)
    # fh.setFormatter(
    #     logging.Formatter(
    #         "%(asctime)s|%(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"
    #     )
    # )
    # logger.addHandler(fh)
    return logger
