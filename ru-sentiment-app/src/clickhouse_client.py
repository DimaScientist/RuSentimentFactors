"""ClickHouse client module."""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import clickhouse_connect
import yake

from config import configurations
from src.enums import Tables
from src.schemas import (
    PredictionResult,
    Summary,
    ShortPrediction,
    SentimentPrediction,
    SentimentFeatures,
    DetailedPredictionData,
    DetailedImageData,
)
from src.text_preprocessing import preprocess_text

if TYPE_CHECKING:
    from typing import List, Optional, Iterator
    from src.schemas import SavedMinioImage


class ClickHouse:
    """ClickHouse client."""

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: Optional[str] = None,
    ):
        self.client_settings = {
            "host": host,
            "port": port,
            "username": username,
        }

        if password:
            self.client_settings["password"] = password

        self.client = clickhouse_connect.get_client(**self.client_settings)

    def get_prediction_by_id(self, prediction_id: uuid.UUID) -> DetailedPredictionData:
        """Get prediction by id."""
        parameters = {"table": Tables.prediction.value, "id": prediction_id}
        query = self.client.query(
            """
            SELECT
            id,
            post_id,
            text,
            predicted_value,
            text_prediction_details_id
            FROM {table:Identifier}
            WHERE id = {id:UUID}
            """,
            parameters=parameters,
        )
        row = query.result_rows[0]

        result = DetailedPredictionData(
            id=row[0],
            post_id=row[1],
            text=row[2],
            predicted_value=row[3],
        )

        if text_prediction_details_id := row[4]:
            text_prediction_details = self.get_prediction_details_by_id(
                text_prediction_details_id,
            )
            result.text_prediction_details = text_prediction_details

        if image_info := self.get_images_by_prediction_id(prediction_id):
            result.image_info = image_info

        return result

    def get_prediction_details_by_id(
        self,
        prediction_details_id: uuid.UUID,
    ) -> SentimentPrediction:
        """Get prediction details by id."""
        parameters = {
            "table": Tables.prediction_details.value,
            "id": prediction_details_id,
        }
        query = self.client.query(
            """
            SELECT
            positive,
            neutral, 
            negative
            FROM {table:Identifier}
            WHERE id = {id:UUID}
            """,
            parameters=parameters,
        )
        row = query.result_rows[0]
        return SentimentPrediction(
            positive=row[0],
            neutral=row[1],
            negative=row[2],
        )

    def get_images_by_prediction_id(
        self,
        prediction_id: uuid.UUID,
    ) -> List[DetailedImageData]:
        """Get images by prediction id."""
        parameters = {"table": Tables.image.value, "prediction_id": prediction_id}
        query = self.client.query(
            """
            SELECT
            id,
            image_url,
            filename,
            prediction_details_id
            FROM {table:Identifier}
            WHERE prediction_id = {prediction_id:UUID}
            """,
            parameters=parameters,
        )
        result = []

        for row in query.result_rows:
            prediction_details_id = row[3]
            prediction_details = self.get_prediction_details_by_id(
                prediction_details_id,
            )
            result.append(
                DetailedImageData(
                    id=row[0],
                    image_url=row[1],
                    filename=row[2],
                    prediction_details=prediction_details,
                )
            )

        return result

    def set_labeled_prediction(self, prediction_id: uuid.UUID, sentiment: str) -> None:
        """Set labeled sentiment to prediction."""
        parameters = {
            "id": prediction_id,
            "labeled_value": sentiment,
            "table": Tables.prediction.value,
        }
        self.client.query(
            """
            ALTER TABLE {table:Identifier} 
            UPDATE labeled_value = {labeled_value:String}
            WHERE id = {id:UUID}
            """,
            parameters=parameters,
        )

    def insert_prediction_details(
        self,
        prediction_details: List[SentimentPrediction],
    ) -> List[uuid.UUID]:
        """Insert prediction details."""
        table = Tables.prediction_details.value
        column_names = ["id", "negative", "neutral", "positive"]

        data = []
        for prediction_detail in prediction_details:
            prediction_details_id = uuid.uuid4()
            prediction_details_row = (
                prediction_details_id,
                prediction_detail.negative,
                prediction_detail.neutral,
                prediction_detail.positive,
            )
            data.append(prediction_details_row)

        self.client.insert(table, data, column_names=column_names)

        return [item[0] for item in data]

    def insert_images(
        self,
        prediction_id: uuid.UUID,
        image_details: List[SentimentPrediction],
        images: List[SavedMinioImage],
    ) -> List[uuid.UUID]:
        """Insert image."""
        table = Tables.image.value
        column_names = [
            "id",
            "image_url",
            "bucket",
            "key",
            "caption",
            "prediction_id",
            "prediction_details_id",
            "filename",
        ]

        prediction_details_ids = self.insert_prediction_details(image_details)

        data = []
        for prediction_details_id, image in zip(prediction_details_ids, images):
            image_id = uuid.uuid4()
            item = (
                image_id,
                image.image_url,
                image.bucket,
                image.key,
                image.caption,
                prediction_id,
                prediction_details_id,
                image.filename,
            )
            data.append(item)

        self.client.insert(table, data, column_names=column_names)

        return [item[0] for item in data]

    def insert_prediction(
        self,
        prediction: PredictionResult,
        post_id: Optional[str] = None,
        text: Optional[str] = None,
        clean_text: Optional[str] = None,
    ) -> uuid.UUID:
        """Insert prediction."""
        table = Tables.prediction.value
        column_names = [
            "id",
            "post_id",
            "text",
            "clean_text",
            "predicted_value",
            "text_prediction_details_id",
        ]

        prediction_id = uuid.uuid4()

        text_prediction_details_id = None
        if text_prediction_details := prediction.prediction_details.text_sentiment:
            prediction_details = [text_prediction_details]
            text_prediction_details_id = self.insert_prediction_details(prediction_details)[0]

        data = [
            (
                prediction_id,
                post_id,
                text,
                clean_text,
                prediction.prediction_result,
                text_prediction_details_id,
            )
        ]

        self.client.insert(table, data, column_names=column_names)

        return prediction_id

    def count(self, table: Tables, table_ids: Optional[List[uuid.UUID]] = None) -> int:
        """Get total row count in table."""
        parameters = {"table": table.value}
        query = "SELECT count() FROM {table:Identifier}"
        if table_ids:
            query += f""" WHERE id IN ({", ".join([f"'{str(table_id)}'" for table_id in table_ids])})"""

        return self.client.command(query, parameters=parameters)

    def get_predictions(self, prediction_ids: Optional[List[uuid.UUID]] = None) -> List[ShortPrediction]:
        """Get predictions."""
        query = """SELECT id, post_id, predicted_value FROM prediction"""

        if prediction_ids:
            query += f"""\nWHERE id IN ({", ".join([f"'{str(prediction_id)}'" for prediction_id in prediction_ids])})"""

        query = self.client.query(query)

        result = []
        for row in query.result_rows:
            result.append(
                ShortPrediction(
                    id=row[0],
                    post_id=row[1],
                    predicted_value=row[2],
                )
            )

        return result

    def prediction_summary(
        self,
        add_features: bool = False,
        add_predictions: bool = False,
        prediction_ids: Optional[List[uuid.UUID]] = None,
    ) -> Summary:
        """Get prediction summary."""
        row_count = self.count(Tables.prediction, prediction_ids)

        parameters = {
            "table": Tables.prediction.value,
            "group_column": "predicted_value",
        }

        query = """
        SELECT
            {group_column:Identifier},
            count() AS predicted_count
        FROM {table:Identifier}
        """

        if prediction_ids:
            query += f"""\nWHERE id IN ({", ".join([f"'{str(prediction_id)}'" for prediction_id in prediction_ids])})"""

        query += "\nGROUP BY {group_column:Identifier}"

        query_result = self.client.query(query, parameters=parameters)

        summary_result = {}
        for row in query_result.result_rows:
            summary_result[row[0]] = row[1] / row_count

        result = Summary(**summary_result)

        if add_features:
            feature_extractor = yake.KeywordExtractor(
                lan="ru",
                n=3,
                dedupLim=0.3,
                top=10,
            )
            sentiment_features = {
                "positive": [],
                "neutral": [],
                "negative": [],
            }

            for sentiment in sentiment_features.keys():
                corpus = self.get_sentiment_corpus(sentiment, True)
                if corpus:
                    corpus = ". ".join(corpus)
                    features = feature_extractor.extract_keywords(corpus)
                    sentiment_features[sentiment] = [item[0] for item in features]

            result.features = SentimentFeatures(**sentiment_features)

        if add_predictions:
            result.predictions = self.get_predictions(prediction_ids)

        return result

    def get_sentiment_corpus(self, sentiment: str, preprocess: bool = True) -> List[str]:
        """Get corpus for sentiment."""
        parameters = {"sentiment": sentiment}

        corpus_query = self.client.query(
            """
            SELECT
                prediction.id,
                prediction.text AS text,
                groupArray(i.caption) AS captions
            FROM prediction
            LEFT JOIN image i on prediction.id = i.prediction_id
            WHERE predicted_value = {sentiment:String}
            GROUP BY prediction.id, prediction.text
            """,
            parameters=parameters,
        )
        corpus = []

        for row in corpus_query.result_rows:
            if text := row[1]:
                corpus.append(text)
            if captions := row[2]:
                corpus.extend(captions)

        if preprocess:
            corpus = preprocess_text(corpus, use_lemmatization=True)

        return corpus


def get_clickhouse() -> Iterator[ClickHouse]:
    """FatAPI dependency for ClickHouse connection."""
    clickhouse = ClickHouse(
        host=configurations.CLICKHOUSE_HOST,
        port=configurations.CLICKHOUSE_PORT,
        username=configurations.CLICKHOUSE_ROOT_USER,
        password=configurations.CLICKHOUSE_ROOT_PASSWORD,
    )
    yield clickhouse
