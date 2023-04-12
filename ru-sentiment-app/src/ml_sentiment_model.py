"""Module for machine learning model."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from models import (
    MultimodalVisualSentimentModel,
    TextSentimentModel,
    bert_tokenizer,
    vit_feature_extractor,
)
from src import schemas
from src.enums import Models
from src.image_preprocessing import extract_features_from_images
from src.text_preprocessing import extract_features_from_text

if TYPE_CHECKING:
    from typing import List, Optional, Union
    from sklearn.preprocessing import LabelEncoder
    from PIL import Image


class MLSentimentModel:
    """Model for sentiment classification."""
    alpha = 0.67

    def __init__(self):
        self.label_encoder: LabelEncoder = Models.label_encoder.load()

        self.bert_tokenizer = bert_tokenizer
        self.visual_feature_extractor = vit_feature_extractor

        self.text_sentiment_model: TextSentimentModel = Models.sentiment_text_model.load()
        self.multimodal_visual_sentiment_model: MultimodalVisualSentimentModel = Models.sentiment_visual_model.load()

    def __form_prediction(
        self, predictions: torch.Tensor
    ) -> Union[List[schemas.ModelSentimentSimplePrediction], schemas.ModelSentimentSimplePrediction,]:
        """Form prediction."""
        predictions_numpy = predictions.cpu().detach().numpy()
        prediction_result = []
        for i in range(len(predictions_numpy)):
            prediction_row = predictions_numpy[i]

            prediction_index = [np.argmax(prediction_row)]

            item_prediction_result = self.label_encoder.inverse_transform(prediction_index)[0]

            prediction_probabilities = {}

            for j in range(len(self.label_encoder.classes_)):
                prediction_probabilities[self.label_encoder.classes_[j]] = prediction_row[j]

            predict_proba = schemas.SentimentPrediction(**prediction_probabilities)

            prediction_result.append(
                schemas.ModelSentimentSimplePrediction(
                    sentiment=item_prediction_result,
                    predict_proba=predict_proba,
                )
            )

        result = prediction_result[0] if len(prediction_result) == 1 else prediction_result
        return result

    def __predict_sentiment_for_images(
        self,
        images: List[Image],
        captions: List[str],
    ) -> List[schemas.ModelSentimentSimplePrediction]:
        """Predict sentiment for images."""
        self.multimodal_visual_sentiment_model.to("cpu")
        self.multimodal_visual_sentiment_model.eval()

        image_features = extract_features_from_images(
            images,
            self.visual_feature_extractor,
            use_image_preprocessing=True,
        )

        input_ids, attention_mask = extract_features_from_text(
            captions,
            self.bert_tokenizer,
            use_lemmatization=True,
        )

        input_ids.to("cpu")
        attention_mask.to("cpu")
        image_features.to("cpu")

        with torch.no_grad():
            predictions = self.multimodal_visual_sentiment_model(input_ids, attention_mask, image_features)

        image_predictions = self.__form_prediction(predictions)
        if type(image_predictions) != list:
            image_predictions = [image_predictions]

        return image_predictions

    def __predict_sentiment_for_text(
        self,
        text: str,
    ) -> schemas.ModelSentimentSimplePrediction:
        """Predict sentiment for text."""
        self.text_sentiment_model.to("cpu")
        self.text_sentiment_model.eval()

        input_ids, attention_mask = extract_features_from_text(
            text,
            self.bert_tokenizer,
            use_lemmatization=True,
        )

        input_ids.to("cpu")
        attention_mask.to("cpu")

        with torch.no_grad():
            predictions = self.text_sentiment_model(
                input_ids,
                attention_mask,
            )
            predictions = predictions

        return self.__form_prediction(predictions)

    def __sentiment_from_float_to_int(self, sentiment: float) -> int:
        """Parse sentiment float value to int."""
        if sentiment < self.alpha:
            sentiment_int = 0
        elif self.alpha <= sentiment < 2 * self.alpha:
            sentiment_int = 1
        else:
            sentiment_int = 2
        return sentiment_int

    def __form_prediction_result(
        self,
        text_predictions: Optional[schemas.ModelSentimentSimplePrediction],
        image_predictions: Optional[List[schemas.ModelSentimentSimplePrediction]],
    ) -> schemas.PredictionResult:
        """Form prediction result."""
        prediction_details = schemas.PredictionDetails()

        if image_predictions is None:
            prediction_details.text_sentiment = text_predictions.predict_proba

            result = schemas.PredictionResult(
                prediction_result=text_predictions.sentiment,
                prediction_details=prediction_details,
            )
        elif text_predictions is None:
            prediction_details.image_sentiment = [prediction.predict_proba for prediction in image_predictions]
            image_predictions_ = [prediction.sentiment for prediction in image_predictions]
            sentiment_labels = self.label_encoder.transform(image_predictions_)
            image_sentiment = [self.__sentiment_from_float_to_int(np.mean(sentiment_labels))]

            result = schemas.PredictionResult(
                prediction_result=self.label_encoder.inverse_transform(image_sentiment)[0],
                prediction_details=prediction_details,
            )
        else:
            prediction_details.text_sentiment = text_predictions.predict_proba
            prediction_details.image_sentiment = [prediction.predict_proba for prediction in image_predictions]

            image_predictions_ = [prediction.sentiment for prediction in image_predictions]

            text_label = self.label_encoder.transform([text_predictions.sentiment])[0]
            image_labels = self.label_encoder.transform(image_predictions_)

            sentiment_results = np.concatenate((np.full_like(image_labels, text_label), image_labels), axis=0)

            summary_result_index = self.__sentiment_from_float_to_int(np.mean(sentiment_results))
            summary_result = self.label_encoder.inverse_transform([summary_result_index])[0]
            result = schemas.PredictionResult(
                prediction_result=summary_result,
                prediction_details=prediction_details,
            )
        return result

    def predict(
        self,
        text: Optional[str] = None,
        images: Optional[List[Image]] = None,
        image_captions: Optional[List[str]] = None,
    ) -> schemas.PredictionResult:
        """Predict sentiment for post."""
        if text is None and images is None:
            raise ValueError("Text or image should be not None.")

        if images and image_captions is None:
            raise ValueError("Images should have captions.")

        text_predictions = None
        if text:
            text_predictions = self.__predict_sentiment_for_text(text)

        image_predictions = None
        if images:
            image_predictions = self.__predict_sentiment_for_images(images, image_captions)

        result_prediction = self.__form_prediction_result(text_predictions, image_predictions)
        return result_prediction
