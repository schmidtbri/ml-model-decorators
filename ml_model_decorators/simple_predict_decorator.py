from ml_base import MLModelDecorator


class SimplePredictDecorator(MLModelDecorator):

    def predict(self, data):
        print("Executing before prediction.")
        prediction = self._model.predict(data=data)
        print("Executing after prediction.")
        return prediction
