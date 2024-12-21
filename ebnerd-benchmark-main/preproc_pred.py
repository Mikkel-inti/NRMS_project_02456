

################################
#
#        PREPROCESSING SUBTITLE
#
###################################


import re

def clean_text_column(text):
    """
    Cleans text by:
    - Removing unwanted characters (excluding dashes between letters).
    - Splitting compound words based on uppercase letters.
    - Removing dashes between numbers.
    - Removing dashes with preceding spaces.
    - Removing extra spaces and trimming the text.


    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.

    Example for subtitle:

    INPUT Text:
    - Jeg har afvist at betale vin til 7000-8000 kr. flasken - det er langt over vores niveauSF bekræfter politianmeldelse mod BrixtofteSkatteyderne betaler for massivt drukHvorfor drikker Brixtofte?Bon-kammeratenLøkke

    OUTPUT Text:
    - Jeg har afvist at betale vin til 7000 8000 kr flasken det er langt over vores niveau SF bekræfter politianmeldelse mod Brixtofte Skatteyderne betaler for massivt druk Hvorfor drikker Brixtofte Bon kammeraten Løkke
    """

    if not isinstance(text, str):  # Check if the input is not a string (e.g., None)
        return text

    # Remove dashes between numbers
    cleaned_text = re.sub(r'(\d)-(\d)', r'\1 \2', text)

    # Remove dashes with preceding spaces
    cleaned_text = re.sub(r'\s-\s', ' ', cleaned_text)

    # Remove unwanted characters except for dashes between letters
    cleaned_text = re.sub(r'[^øåæ\w\s-]', ' ', cleaned_text)

    # Add space between lowercase and uppercase letters
    cleaned_text = re.sub(r'(?<=[a-zøåæ])(?=[A-Z])', ' ', cleaned_text)

    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text




################################
#
#           ADD PREDICTION SCORES
#
###################################
from typing import Iterable
import polars as pl
from ebrec.utils._polars import generate_unique_name
from ebrec.utils._constants import DEFAULT_INVIEW_ARTICLES_COL
import numpy as np


def add_prediction_scores(
    df: pl.DataFrame,
    scores: Iterable[float],
    prediction_scores_col: str = "scores",
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
) -> pl.DataFrame:
    # Generate a unique ID for groupby
    GROUPBY_ID = generate_unique_name(df.columns, "_groupby_id")

    # Add row index as GROUPBY_ID in the original DataFrame
    df = df.with_row_index(name=GROUPBY_ID)

    # Generate prediction scores
    scores = (
        df.lazy()
        .select(pl.col(inview_col), pl.col(GROUPBY_ID))
        .explode(inview_col)
        .with_columns(pl.Series(prediction_scores_col, scores).explode())
        .group_by(GROUPBY_ID)
        .agg(inview_col, prediction_scores_col)
        .sort(GROUPBY_ID)
        .collect()
    )

    return (
        df.join(scores, on=GROUPBY_ID, how="left")
        .drop(GROUPBY_ID)
    )
