from preprocess import Preprocessor
from eval import Evaluator
from postprocess import Postprocessor
import pickle
        
class Pipeline:
    def __init__(self, dev_set: bool = False, config = configuration):
        self.dev_set = dev_set
        self.preprocessor = Preprocessor()
        # self.model = config[model]?
        self.evaluation = {}

    # run performs a full step of the pipe line, it loads in the clean data the way the configuration wants it. 
    def run(self):
        # load, clean and the data
        X, y = self.preprocessor.load_data()
        
        # fit the model
        # TODO: also takes care of the train test split? 
        self.model.fit(X, y)
        # make a prediction:
        y_pred = self.model.predict(X, y)

        if(!self.dev_set):
        # eval the prediction
        evaluator = Evaluator(y_pred=y_pred, y_true=y_true)
        self.evaluation = evaluator.eval()

        # write out the predictions to an external file
        postprocessor = Postprocessor(y_pred)
        self.postprocessor.save_predictions()

