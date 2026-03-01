# **Word2Vec from Scratch**

A pure NumPy implementation of Word2Vec covering both Skip-gram and CBOW.
No machine learning frameworks are used.

## **What is implemented**

* Skip-gram Word2Vec with negative sampling
* CBOW Word2Vec
* Unigram noise distribution raised to the 3/4 power
* Subsampling of frequent
* Dynamic window size, sampled per center word each epoch
* Linear learning rate decay
* Minimum word count filtering

## **Running**
```
python main.py
```
This trains Skip-gram first, then CBOW, each for 5 epochs on the Brown corpus.
Both sets of embeddings are saved in the word2vec binary format.

* vectors_sg.bin . . . Skip-gram embeddings
* vectors_cbow.bin . . . CBOW embeddings

## **References**

Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean.
Efficient Estimation of Word Representations in Vector Space.
arXiv 1301.3781, 2013.

Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean.
Distributed Representations of Words and Phrases and their Compositionality.
arXiv 1310.4546, 2013.
