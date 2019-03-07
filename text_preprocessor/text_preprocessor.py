import re
from functools import lru_cache
from ekphrasis.classes.spellcorrect import SpellCorrector
from ekphrasis.classes.segmenter import Segmenter
from text_preprocessor.regular_expressions import regexes
from text_preprocessor.dictionaries import emoticons


class TextPreProcessor:
    """
    Kwargs:
        normalize (list)
            possible values: ['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date']

        annotate (list)
            possible values: ['hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored']

        unpack_hashtags (bool)

        unpack_contractions (bool)

        segmenter (str): define the statistics of what corpus you would
            like to use [english, twitter]

        corrector (str): define the statistics of what corpus you would
            like to use [english, twitter]

        tokenizer (callable): callable function that accepts a string and
                returns a list of strings if no tokenizer is provided then
                the text will be tokenized on whitespace

        simplify_emoticons (bool)

        dictionaries (list)
    """

    def __init__(self, **kwargs):
        self.tokens_to_normalize = kwargs.get("normalize", [])
        self.annotate = kwargs.get("annotate", [])
        self.unpack_hashtags = kwargs.get("unpack_hashtags", False)
        self.unpack_contractions = kwargs.get("unpack_contractions", False)
        self.segmenter_corpus = kwargs.get("segmenter", "english")
        self.corrector_corpus = kwargs.get("corrector", "english")
        self.segmenter = Segmenter(corpus=self.segmenter_corpus)
        self.spell_corrector = SpellCorrector(corpus=self.corrector_corpus)
        self.tokenizer = kwargs.get("tokenizer", None)
        self.simplify_emoticons = kwargs.get("simplify_emoticons", False)
        self.dictionaries = kwargs.get("dictionaries", [])
        self.stats = {}
        self.preprocessed_texts = -1

    def pre_process(self, text: str, with_stats=False):
        self._increment_counter()

        text = self._remove_repeating_spaces(text)
        text = self._normalize(text)
        text = self._unpack_hashtags(text)
        text = self._annotate(text)
        text = self._unpack_contractions(text)
        text = self._remove_repeating_spaces(text)

        tokens = self._tokenize(text)
        tokens = self._simplify_emoticons(tokens)
        tokens = self._replace_using_dictionaries(tokens)

        if with_stats:
            return tokens, self._pre_processed_text_stats()
        else:
            return tokens

    def _pre_processed_text_stats(self):
        return self.stats[self.preprocessed_texts]

    def _increment_counter(self):
        self.preprocessed_texts += 1
        self.stats[self.preprocessed_texts] = {}

    def _normalize(self, text):
        for item in self.tokens_to_normalize:
            text = self._change_using_regexp(item, lambda m: f' <{item}> ', text, 'normalize')
        return text

    def _unpack_hashtags(self, text):
        if self.unpack_hashtags:
            return self._change_using_regexp("hashtag", lambda w: self._handle_hashtag_match(w), text, "unpack")
        return text

    def _annotate(self, text):
        text = self._annotate_allcaps(text)
        text = self._annotate_elongated(text)
        text = self._annotate_repeated(text)
        text = self._annotate_emphasis(text)
        text = self._annotate_censored(text)
        return text

    def _annotate_allcaps(self, text):
        if "allcaps" in self.annotate:
            return self._change_using_regexp("allcaps", lambda w: self._handle_generic_match(w, "allcaps", mode='wrap'),
                                             text, "annotate")
        return text

    def _annotate_elongated(self, text):
        if "elongated" in self.annotate:
            return self._change_using_regexp("elongated", lambda w: self._handle_elongated_match(w), text, "annotate")
        return text

    def _annotate_repeated(self, text):
        if "repeated" in self.annotate:
            return self._change_using_regexp("repeat_puncts", lambda w: self._handle_repeated_puncts(w), text,
                                             "annotate")
        return text

    def _annotate_emphasis(self, text):
        if "emphasis" in self.annotate:
            return self._change_using_regexp("emphasis", lambda w: self._handle_emphasis_match(w), text, "annotate")
        return text

    def _annotate_censored(self, text):
        if "censored" in self.annotate:
            return self._change_using_regexp("censored", lambda w: self._handle_generic_match(w, "censored"), text,
                                             "annotate")
        return text

    def _change_using_regexp(self, regexp_name, func, text, stats_name_prefix):
        changing_result = regexes[regexp_name].subn(func, text)
        self._update_stats(f'{stats_name_prefix}_{regexp_name}', changing_result[1])
        return changing_result[0]

    def _unpack_contractions(self, text):
        if self.unpack_contractions:
            text = self._unpack_selected_contrations(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|"
                                                     r"[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n'?t",
                                                     r"\1\2 not", text)

            text = self._unpack_selected_contrations(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll",
                                                     r"\1\2 will", text)
            text = self._unpack_selected_contrations(r"(\b)([Tt]hey|[Ww]hat|[Ww]ho|[Yy]ou)ll", r"\1\2 will", text)

            text = self._unpack_selected_contrations(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
            text = self._unpack_selected_contrations(r"(\b)([Tt]hey|[Ww]hat|[Yy]ou)re", r"\1\2 are", text)

            text = self._unpack_selected_contrations(r"(\b)([[Hh]e|[Ss]he)'s", r"\1\2 is", text)

            text = self._unpack_selected_contrations(
                r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)"
                r"'?ve", r"\1\2 have", text)

            text = self._unpack_selected_contrations(r"(\b)([Cc]a)n't", r"\1\2n not", text)
            text = self._unpack_selected_contrations(r"(\b)([Ii])'m", r"\1\2 am", text)
            text = self._unpack_selected_contrations(r"(\b)([Ll]et)'?s", r"\1\2 us", text)
            text = self._unpack_selected_contrations(r"(\b)([Ww])on'?t", r"\1\2ill not", text)
            text = self._unpack_selected_contrations(r"(\b)([Ss])han'?t", r"\1\2hall not", text)
            text = self._unpack_selected_contrations(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)

        return text

    def _unpack_selected_contrations(self, regexp, replacement, text):
        unpacking_result = re.subn(regexp, replacement, text)
        self._update_stats("unpack_contrations", unpacking_result[1])
        return unpacking_result[0]

    def _tokenize(self, text):
        if self.tokenizer:
            return self.tokenizer(text)
        else:
            return text.split(' ')

    def _simplify_emoticons(self, tokens):
        if self.simplify_emoticons:
            result = []
            for token in tokens:
                if token in emoticons:
                    new_emoticon = emoticons[token]
                    if new_emoticon != token:
                        self._update_stats('emoticon_simplification', 1)
                    result.append(new_emoticon)
                else:
                    result.append(token)
            return result
        else:
            return tokens

    def _replace_using_dictionaries(self, tokens):
        if len(self.dictionaries) > 0:
            for dictionary in self.dictionaries:
                for idx, token in enumerate(tokens):
                    if token in dictionary:
                        value = dictionary[token]
                        if '<entity>' not in value:
                            tokens[idx] = value
                            self._update_stats('dictionary_replacement', 1)
            return ' '.join(tokens).split(' ')
        else:
            return tokens

    @lru_cache(maxsize=65536)
    def _handle_hashtag_match(self, m):
        text = m.group()[1:]

        if text.islower():
            expanded = self.segmenter.segment(text)
            expanded = " ".join(expanded.split("-"))
            expanded = " ".join(expanded.split("_"))
        else:
            expanded = regexes["camel_split"].sub(r' \1', text)
            expanded = expanded.replace("-", "")
            expanded = expanded.replace("_", "")

        if "hashtag" in self.annotate:
            expanded = self._add_special_tag(expanded, "hashtag", mode="wrap")

        return expanded

    @lru_cache(maxsize=65536)
    def _handle_generic_match(self, m, tag, mode="every"):
        text = m.group()
        if tag == 'allcaps':  # word around for allcaps contractions like YOU'RE TODO refactor
            text = text.lower()

        text = self._add_special_tag(text, tag, mode=mode)

        return text

    def _handle_elongated_match(self, m):
        text = m.group()

        text = regexes["normalize_elong"].sub(r'\1\1', text)

        normalized = self.spell_corrector.normalize_elongated(text)
        if normalized:
            text = normalized

        text = self._add_special_tag(text, "elongated")

        return text

    @lru_cache(maxsize=65536)
    def _handle_repeated_puncts(self, m):
        text = m.group()
        text = "".join(sorted(set(text), reverse=True))
        text = self._add_special_tag(text, "repeated")

        return text

    @lru_cache(maxsize=65536)
    def _handle_emphasis_match(self, m):
        text = m.group().replace("*", "")
        text = self._add_special_tag(text, "emphasis")

        return text

    def _update_stats(self, key, value):
        if value > 0:
            stats_for_text = self.stats[self.preprocessed_texts]

            if key not in stats_for_text:
                stats_for_text[key] = 0
            stats_for_text[key] += value

    @staticmethod
    def _remove_repeating_spaces(text):
        return re.sub(r' +', ' ', text).strip()

    @staticmethod
    def _add_special_tag(m, tag, mode="single"):

        if isinstance(m, str):
            text = m
        else:
            text = m.group()

        if mode == "single":
            return " {} <{}> ".format(text, tag)
        elif mode == "wrap":
            return " ".join([" <{}> {} </{}> ".format(tag, text, tag)]) + " "
        elif mode == "every":
            tokens = text.split()
            processed = " ".join([" {} <{}> ".format(t, tag)
                                  for t in tokens])
            return " " + processed + " "
