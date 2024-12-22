from preprocess import Preprocessor
from eval import Evaluator, ErrorAnalyzer
from postprocess import Postprocessor
import pickle
        
class Pipeline:
    # initialize the pipeline with a preprocessor, model and a postprocessor
    def __init__(self, preprocessor: Preprocessor, model): #, postprocessor: Postprocessor
        self.preprocessor = preprocessor
        self.model = model # either pass one of the model classes or load from a pkl file for the devset predictions
    
    # run performs a full step of the pipe line, it loads in the clean data the way the configuration wants it. 
    def run(self):
        # load, clean and split? the data
        X, y, ids = self.preprocessor.preprocess()
        print("PREPROCESSED THE DATA")
        
        # if there is train data, train the model on the train data:
        if self.preprocessor.split:
            print("TRAINING THE MODEL")
            self.model.fit(X["train"], y["train"])
            print("FIT THE DATA")
            
        
        # make a prediction: (In the preprocessr)
        print("MAKING PREDICTIONS")
        self.y_pred = self.model.predict(X["test"])
        print("COMPLETED PREDICTIONS")

        # once we have the prediction, we want to evaluate how good the prediction is:  
        evaluator = Evaluator(self.y_pred, y["test"])
        self.evaluation = evaluator.eval()
        print("COMPLETED EVAL")

        # we also want to analyse for errors: what does the model mispredict?
        self.error_analyzer = ErrorAnalyzer(y_pred=self.y_pred, y_true=y["test"], ids=ids["test"])
        self.error_analyzer.analyze()
        
        # lastly, write out the predictions to an external file
        postprocessor = Postprocessor(self.y_pred, ids["test"])
        postprocessor.save_predictions(model_name=self.model.name)

        
        