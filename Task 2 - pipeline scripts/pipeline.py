from preprocess import Preprocessor
from eval import Evaluator
from postprocess import Postprocessor
class Pipeline:
    def __init__(self):
        self.preprocessor = Preprocessor()
        # self.model = config[model]
        self.evaluator = Evaluator()
        self.postprocessor = Postprocessor()

    def run(self):
        df = self.preprocessor.load_data()

        # X/y?
        # fit the model
        self.model.fit(X, y)

        # make a prediction:
        y_pred = self.model.predict(X, y)

        # 