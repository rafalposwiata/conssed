# ConSSED

Repository for Configurable Semantic and Sentiment Emotion Detector (ConSSED) - our system participating in the SemEval-2019 Task 3: EmoContext: Contextual Emotion Detection in Text (https://www.humanizing-ai.com/emocontext.html).

Publication:

[ConSSED at SemEval-2019 Task 3: Configurable Semantic and Sentiment Emotion Detector](https://www.aclweb.org/anthology/S19-2027)

Citation:
```bash
@InProceedings{ConSSED-2019,
  author    = {Poświata, Rafał},
  title     = {ConSSED at SemEval-2019 Task 3: Configurable Semantic and Sentiment Emotion Detector},
  booktitle = {Proceedings of the 13th International Workshop on Semantic Evaluation (SemEval-2019)},
  year = {2019},
  pages = {175–179}
}
```

## Installation steps

#### 1. Download the repository
```bash
git clone https://github.com/rafalposwiata/conssed.git
```

#### 2. Create docker image
```bash
docker build -t conssed .
```
#### 3. Download resources

Resources necessary for reconstructing the results from the publication are available [here](https://drive.google.com/open?id=1jna0wnCBR61nCTKG7qSfhEnN4k_SqYsz).

#### 4. Complete resources

Due to the fact that some of the resources we used are protected by certain restrictions, we could not add them to the resources folder.
In order to use the ConSSED system, two types of missing resources must be filled in: data and embeddings. Data shared by EmoContext organizers (train.txt, dev.txt and test.txt files) should be added to the data folder.
Embeddings should be added according to the table below.


| Embedding  | File  | Source link | Destination directory  | 
|---|---|---|---|
| GloVe      | glove.twitter.27B.100d.txt |http://nlp.stanford.edu/data/glove.twitter.27B.zip | resources/embeddings/glove  |
| NTUA_310      | ntua_twitter_affect_310.txt |https://drive.google.com/open?id=1b-w7xf0d4zFmVoe9kipBHUwfoefFvU2t | resources/embeddings/word2vec  |
| SSWE      | sswe-r.txt |http://ir.hit.edu.cn/~dytang/paper/sswe/embedding-results.zip | resources/embeddings/sswe  |
| Emo2Vec      | emo2vec.txt |https://drive.google.com/file/d/1K0RPGSlBHOng4NN4Jkju_OkYtrmqimLi/view?usp=sharing | resources/embeddings/emo2vec  |


## Reconstruction of the results from the publication

```bash
docker run --runtime=nvidia -v /path-to-resources-directory/:/resources conssed python3.6 /conssed/predict.py /resources/models/<model_name>/predict.config
```

Where <model_name> is one of the trained models, the list of which is as follows:
 - BiLSTM_GloVe
 - BiLSTM_ELMo
 - BiLSTM_NTUA_310
 - BiLSTM_SSWE
 - BiLSTM_Emo2Vec
 - ConSSED_GloVe_SSWE
 - ConSSED_GloVe_Emo2Vec
 - ConSSED_ELMo_SSWE
 - ConSSED_ELMo_Emo2Vec
 - ConSSED_NTUA_310_SSWE
 - ConSSED_NTUA_310_Emo2Vec
 - ConSSED_NTUA_310_Emo2Vec_v2

## Train new models

TODO
