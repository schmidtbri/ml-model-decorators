from pydantic import BaseModel, Field
from enum import Enum
from ml_base.ml_model import MLModel


class ModelInput(BaseModel):
    sepal_length: float = Field(gt=5.0, lt=8.0)
    sepal_width: float = Field(gt=2.0, lt=6.0)
    petal_length: float = Field(gt=1.0, lt=6.8)
    petal_width: float = Field(gt=0.0, lt=3.0)


class Species(str, Enum):
    iris_setosa = "Iris setosa"
    iris_versicolor = "Iris versicolor"
    iris_virginica = "Iris virginica"


class ModelOutput(BaseModel):
    species: Species


class IrisModelMock(MLModel):
    display_name = "Iris Model"
    qualified_name = "iris_model"
    description = "A model to predict the species of a flower based on its measurements."
    version = "1.0.0"
    input_schema = ModelInput
    output_schema = ModelOutput

    def __init__(self) -> None:
        pass

    def predict(self, data: ModelInput) -> ModelOutput:
        return ModelOutput(species="Iris setosa")