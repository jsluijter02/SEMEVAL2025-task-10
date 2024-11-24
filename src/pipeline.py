from preprocess import Preprocessor
from eval import Evaluator
from postprocess import Postprocessor
import pickle
        
class Pipeline:
    # initialize the pipeline with a preprocessor, model and a postprocessor
    def __init__(self, preprocessor: Preprocessor, model): #, postprocessor: Postprocessor
        self.preprocessor = preprocessor
        self.model = model # either pass one of the model classes or load from a pkl file for the devset predictions
        #self.postprocessor = postprocessor
    
    # run performs a full step of the pipe line, it loads in the clean data the way the configuration wants it. 
    def run(self):
        # load, clean and split? the data
        X, y = self.preprocessor.preprocess()
        print("PREPROCESSED THE DATA")
        
        # if there is train data, train the model on the train data:
        if self.preprocessor.split:
            self.model.fit(X["train"], y["train"])
            print("FIT THE DATA")
            # make a prediction:
            print("MAKING PREDICTIONS")
            self.y_pred = self.model.predict(X["test"])
            print("COMPLETED PREDICTIONS")
            
            evaluator = Evaluator(self.y_pred, y["test"])
            self.evaluation = evaluator.eval()
            print("COMPLETED EVAL")
        
        # if we only have test data, just make the prediction
        else:
            self.y_pred = self.model.predict(X)

        # TODO: figure out how to keep the frigging ids
        # # write out the predictions to an external file
        # postprocessor = Postprocessor(self.y_pred)
        # postprocessor.save_predictions()

        
        