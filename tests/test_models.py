"""Model smoke tests."""

from pathlib import Path

from medtabx.data import Dataset
from medtabx.data.io_tabular import load_csv
from medtabx.encoders import EmbeddingEncoder, LinearEncoder
from medtabx.models.ft_transformer import FTTransformer
import medtabx.core.stype as stype


def test_ft_transformer_smoke():
    data_root = Path(__file__).parent / "data"
    if not data_root.exists():
        data_root = Path(__file__).parent.parent / "examples" / "data"

    csv_path = data_root / "tabular" / "diabetes_sample.csv"
    df = load_csv(str(csv_path))

    col_to_stype = {
        "patient_id": stype.categorical,
        "age": stype.numerical,
        "gender": stype.categorical,
        "bmi": stype.numerical,
        "blood_pressure": stype.numerical,
        "glucose": stype.numerical,
        "insulin": stype.numerical,
        "diabetes": stype.categorical,
    }

    dataset = Dataset(df=df, col_to_stype=col_to_stype, target_col="diabetes")
    dataset.materialize()

    stype_encoder_dict = {
        stype.numerical: LinearEncoder(),
        stype.categorical: EmbeddingEncoder(),
    }

    model = FTTransformer(
        channels=32,
        out_channels=2,
        num_layers=2,
        col_stats=dataset.col_stats,
        col_names_dict=dataset.tensor_frame.col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
    )

    out = model(dataset.tensor_frame)
    assert out.shape[0] == len(df)
    assert out.shape[1] == 2
