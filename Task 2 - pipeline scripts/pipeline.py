from preprocess import Preprocessor
from eval import Evaluator
from postprocess import Postprocessor
import pickle
        
class Pipeline:
    # initialize the pipeline with a preprocessor and model
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model # either pass one of the model classes or load from a pkl file for the devset predictions

    # run performs a full step of the pipe line, it loads in the clean data the way the configuration wants it. 
    def run(self, devset:bool = False):
        if not devset:
            # load, clean and split the data
            X, y = self.preprocessor.load_data()
            # fit the model
            # TODO: also takes care of the train test split? 
            self.model.fit(X["train"], y["train"])

        else:
            X = self.preprocessor.load_dev_data()

        # clean data?
        self.preprocessor.clean(X) # clean X and transform in for instance 

        # make a prediction:
        self.y_pred = self.model.predict(X, y)

        # write out the predictions to an external file
        postprocessor = Postprocessor(self.y_pred)
        self.postprocessor.save_predictions()

    def eval(self, y_pred, y_true):
        # eval the prediction
        evaluator = Evaluator(y_pred=y_pred, y_true=y_true)
        self.evaluation = evaluator.eval()