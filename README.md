# INF8225-TP4
A word2vec implementation in python of the Continuous Bag of Words (CBOW) and Skip-gram neural network architectures using Hierarchical Softmax and Negative Sampling learning algorithms for efficient learning of word vectors (Mikolov, et al., 2013a, b, c; http://code.google.com/p/word2vec/).

## Usage
- Install python 3.6 or higher and install dependencies from requirements.txt
```
    pip3 install -r requirements.txt
```

- Train and test word vectors:
```
word2vec.py [-h] [-test TEST] -model FO [-train FI] [-cbow CBOW]
                   [-negative NEG] [-dim DIM] [-alpha ALPHA] [-window WIN]
                   [-min-count MIN_COUNT] [-processes NUM_PROCESSES]
                   [-epochs EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  -test TEST            Load trained model and test it
  -model FO             Output model file -> if test is enabled, then provide
                        the path for the model to test
  -train FI             Training file
  -cbow CBOW            1 for CBOW, 0 for skip-gram
  -negative NEG         Number of negative examples (>0) for negative
                        sampling, 0 for hierarchical softmax
  -dim DIM              Dimensionality of word embeddings
  -alpha ALPHA          Starting alpha
  -window WIN           Max window length
  -min-count MIN_COUNT  Min count for words used to learn <unk>
  -processes NUM_PROCESSES
                        Number of processes
  -epochs EPOCHS        Number of training epochs
```

## Evaluation
We used this implementation for testing the approch of the two research papers on smaller training sets and with the minumum of epochs possible to obtain an acceptable result. We decided to train on 16 millions wikipedia words rather than 15 billions and we stopped in 3 epochs for the algorithms. Here are our results :


## References
Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013a). Distributed representations of words and phrases and their compositionality. Advances in Neural Information Processing Systems. http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013b). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781. http://arxiv.org/pdf/1301.3781.pdf