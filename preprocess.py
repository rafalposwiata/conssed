from ekphrasis.classes.tokenizer import SocialTokenizer
from text_preprocessor.text_preprocessor import TextPreProcessor
from text_preprocessor.dictionaries import abbreviations, common_mistakes

default_text_pre_processor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date'],
    annotate=['hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored'],
    unpack_hashtags=True,
    unpack_contractions=True,
    simplify_emoticons=True,
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dictionaries=[abbreviations, common_mistakes]
)

simple_text_pre_processor = TextPreProcessor(
    unpack_contractions=True,
    simplify_emoticons=True,
    tokenizer=SocialTokenizer().tokenize,
    dictionaries=[abbreviations, common_mistakes]
)


def get_pre_processor(pre_processing: str):
    if pre_processing == 'default':
        return default_text_pre_processor
    elif pre_processing == 'simple':
        return simple_text_pre_processor


def pre_process(lines, pre_processing: str):
    pre_processor = get_pre_processor(pre_processing)

    result = []
    for line in lines:
        tokens = []
        for utterance in line[1:4]:
            tokens += pre_processor.pre_process(utterance)
            tokens.append('eos')
        result.append(tokens[:-1])
    return result
