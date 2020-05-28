from cade.cade import CADE
from gensim.models.word2vec import Word2Vec

from preprocess import create_compass

# Create compass
create_compass(
    [
        "./data/english/semeval2020_ulscd_eng/corpus1/lemma/ccoha1.txt",
        "./data/english/semeval2020_ulscd_eng/corpus2/lemma/ccoha2.txt",
    ],
    save_path="compass.txt",
)

aligner = CADE(size=30)

# train the compass: the text should be the concatenation of the text from the slices
aligner.train_compass("compass.txt", overwrite=True)

# now you can train slices and they will be already aligned
# these are gensim word2vec objects
slice_one = aligner.train_slice(
    "./data/english/semeval2020_ulscd_eng/corpus1/lemma/ccoha1.txt", save=True
)
slice_two = aligner.train_slice(
    "./data/english/semeval2020_ulscd_eng/corpus2/lemma/ccoha2.txt", save=True
)
# model1 = Word2Vec.load("model/arxiv_14.model")
# model2 = Word2Vec.load("model/arxiv_9.model")
