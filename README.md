# ConSSED

Repository for Configurable Semantic and Sentiment Emotion Detector (ConSSED) - our system participating in the SemEval-2019 Task 3: EmoContext: Contextual Emotion Detection in Text (https://www.humanizing-ai.com/emocontext.html).

## Installation Steps

#### 1. Download the repository
```bash
git clone https://github.com/rafalposwiata/conssed.git
```

#### 2. Create docker image
```bash
docker build -t conssed .
```
#### 3. Download resources

Resources necessary for reconstructing the results from the publication are available[here](https://drive.google.com/open?id=1jna0wnCBR61nCTKG7qSfhEnN4k_SqYsz).

#### 4. Complete resources

Due to the fact that some of the resources we used are protected by certain restrictions, we could not add them to the resources folder.
In order to use the ConSSED system, two types of missing resources must be filled in: data and embeddings. Data shared by organizers (train.txt, dev.txt and test.txt files) should be added to the date folder.
Embeddings should be added according to the table below.


| Embedding  | File  | Source link | Destination directory  | 
|---|---|---|---|
| GloVe      | glove.twitter.27B.100d.txt |http://nlp.stanford.edu/data/glove.twitter.27B.zip | resource/embeddings/glove  |
| NTUA_310      | ntua_twitter_affect_310.txt |https://drive.google.com/open?id=1b-w7xf0d4zFmVoe9kipBHUwfoefFvU2t | resource/embeddings/word2vec  |
| SSWE      | sswe-r.txt |http://ir.hit.edu.cn/~dytang/paper/sswe/embedding-results.zip | resource/embeddings/sswe  |
| Emo2Vec      | emo2vec.txt |https://drive.google.com/file/d/1K0RPGSlBHOng4NN4Jkju_OkYtrmqimLi/view?usp=sharing | resource/embeddings/emo2vec  |


## Reconstruction of the results from the publication.

```bash
docker run -v /path-to-resources-directory/:/resources conssed python3.6 /conssed/predict.py /resources/models/<model_name>/predict.config
```

Where <model_name> is one of trained model for example: BiLSTM_GloVe or ConSSED_NTUA_310_Emo2Vec.

## Train new models

TODO
