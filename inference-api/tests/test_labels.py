"""Метки в схемах совпадают с app.labels."""

from typing import get_args

from app import model, schemas
from app.labels import CLASS_IDS, LABEL_IDS, LABELS, N_CLASSES


def test_labels_constants_are_consistent():
    assert CLASS_IDS == tuple(LABELS) and N_CLASSES == len(LABELS) == 3
    assert {LABELS[idx]: idx for idx in CLASS_IDS} == LABEL_IDS
    assert model.LABELS is LABELS


def test_schemas_match_labels():
    assert set(get_args(schemas.Label)) == set(LABELS.values())
    assert set(schemas.Scores.model_fields) == set(LABELS.values())
    label_id = schemas.PredictResponse.model_fields["label_id"].annotation
    assert set(get_args(get_args(label_id)[0])) == set(CLASS_IDS)
