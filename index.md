# ELMo Word Embedding
## Introduction

ELMo is created by AllenNLP which provides the contextualized word embeddings whose vector representation for a word differs in a sentence to sentence.

Now let’s see the real-life application of EMLo embedding, suppose  you are looking for ‘sony’ on goolge then it predicts 'liv'and 'six liv' in order to make the result more precise and now you would be able to get what you want thus making our lives more easier.

ELMo is a deep contextualized word representation that models both complex characteristics of word use, and how these uses vary across linguistic contexts. These word vectors are learned functions of the internal states of a deep bidirectional language model, which is pre-trained on a large text corpus.
## What is Word Embedding?

A word embedding is a learned representation for text where words that have the same meaning have a similar representation.

It is this approach to representing words and documents that may be considered one of the key breakthroughs of deep learning on challenging natural language processing problems.\
\
"*One of the benefits of using dense and low-dimensional vectors is computational: the majority of neural network toolkits do not play well with very high-dimensional, sparse vectors. … The main benefit of the dense representations is generalization power: if we believe some features may provide similar clues, it is worthwhile to provide a representation that is able to capture these similarities*."\
\
**-- Page 92, Neural Network Methods in Natural Language Processing, 2017.
In simple language:

Word embeddings are in fact a class of techniques where individual words are represented as real-valued vectors in a predefined vector space. Each word is mapped to one vector and the vector values are learned in a way that resembles a neural network, and hence the technique is often lumped into the field of deep learning.
## Is Word Embedding useful?
Let us assume we have a corpus with vocab of 10000 words, now to represent them simply we can do one-hot encoding i.e Each word will be represented as an n-dimensional vector, where n is the vocabulary size
Each word’s vector representation will be mostly “0”, except there will be a single “1” entry in the position corresponding to the word’s index in the vocabulary.
Now suppose we want to translate the English input sentence “the cat is black” into another language, now we create one-hot encoding of the sentence.
this process has generated a very sparse (mostly zero) feature vector for each input word
the end results can be mediocre, especially when the training dataset is small. This is because one-hot vectors aren’t a great input representation method
The similarity issue. Ideally we would want similar words like “cat” and “tiger” to have somewhat similar features. But with these one-hot vectors, “cat” is as similar to “tiger” as literally any other word, which isn’t great
The vocabulary size issue: One-hot vector dimensionality is the same as number of words. There’s reasons why you don’t want your feature size to explode —namely, more features means more parameters to estimate, and you require exponentially more data to estimate those parameters well enough to build a reasonably generalisable model
The computational issue. Each word’s embedding/feature vector is mostly zeroes, and many machine learning models won’t work well with very high dimensional and sparse features. Neural networks in particular struggle with this type of data (though there’s workarounds, e.g. using a type of LASSO-like feature selection). With such a large feature space, you are also in danger of running into memory and even storage concerns, especially if the models you’re working with don’t play nicely with compressed versions of sparse matrices

The core problem that embeddings solve is generalisation.

This means embeddings allow us to build much more generalisable models–instead of the network needing to scramble to learn many disparate ways to handle disconnected input, we instead let similar words “share” parameters and computation paths.
If we take 5 example words from our vocabulary (say… the words “aardvark”, “black”, “cat”, “duvet” and “zombie”) and examine their embedding vectors created by the one-hot encoding method discussed above, the result would look like this:
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Abeast-1651/Online_Content/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.