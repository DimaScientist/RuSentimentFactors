"""Module for text preprocessing."""
from __future__ import annotations

import re

from typing import TYPE_CHECKING

import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import RussianStemmer
from nltk.tokenize.casual import casual_tokenize
from pymystem3 import Mystem
import torch

if TYPE_CHECKING:
    from typing import List, Tuple, Union

    from transformers import AutoTokenizer

LANGUAGE = "russian"
nltk.download("stopwords")
LANG_STOPWORDS = set(stopwords.words(LANGUAGE))
mystem = Mystem()
stemmer = RussianStemmer()


def clean_sentence(sentence: str) -> str:
    """Clean sentence."""
    text = BeautifulSoup(sentence, "lxml").get_text()
    letters_only = re.sub("[^а-яА-ЯёЁ]", " ", text)

    lowercase_text = letters_only.lower()

    string_without_stopwords = " ".join(
        [word for word in casual_tokenize(lowercase_text) if word not in LANG_STOPWORDS]
    )
    return string_without_stopwords


def sentence_lemmatization(sentence: str) -> str:
    """Lemmatize sentence."""
    lemmatize_words = mystem.lemmatize(sentence)
    return "".join(lemmatize_words).strip()


def sentence_stemming(sentence: str) -> str:
    """Apply stemmer to sentence."""
    words = casual_tokenize(sentence)
    stemmer_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmer_words)


def preprocess_text(
    text: Union[str, List[str]],
    use_lemmatization: bool = False,
    use_stemming: bool = False,
) -> Union[str, List[str]]:
    """Preprocess text."""
    text_ = text

    if type(text_) != list:
        text_ = [text_]

    preprocessed_text = []

    for sentence in text_:
        cleaned_sentence = clean_sentence(sentence)

        if use_stemming:
            cleaned_sentence = sentence_stemming(cleaned_sentence)
            preprocessed_text.append(cleaned_sentence)

    if use_lemmatization:
        text_ = preprocessed_text or text_
        united_text = "\n".join(text_)
        preprocessed_text = sentence_lemmatization(united_text).split("\n")

    result = preprocessed_text[0] if len(preprocessed_text) == 1 else preprocessed_text
    return result


def extract_features_from_text(
    text: Union[List[str], str],
    tokenizer: AutoTokenizer,
    clean_text: bool = True,
    use_lemmatization: bool = False,
    use_stemming: bool = False,
    sequence_length: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features from text."""
    text_ = text

    if clean_text:
        text_ = preprocess_text(text_, use_lemmatization, use_stemming)

    if type(text_) != list:
        text_ = [text_]

    input_ids = []
    attention_masks = []
    for sentence in text_:
        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            truncation=True,
            max_length=sequence_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids.append(encoded_dict.get("input_ids"))
        attention_masks.append(encoded_dict.get("attention_mask"))

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)
