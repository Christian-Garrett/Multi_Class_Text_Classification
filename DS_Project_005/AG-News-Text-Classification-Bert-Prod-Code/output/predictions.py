from pathlib import Path
import sys
import os
import pandas as pd
import warnings
warnings.simplefilter(action='ignore')

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))

from src.ML_Pipeline import model


try:
    # Load Fine-tuned Bert Model for further predictions:
    print('##### Load Fine-tuned Bert Model for further predictions #####')
    bert_predictor = model.load_model()

except Exception as e:
    print('!! Exception Details: !!\n', e.__class__)
    print('Please debug for further details')


# Two articles taken from the Associated Press were correctly classified for validation
test_data = pd.read_csv(os.path.join(module_path, "input/test_data.csv"), 
                      usecols=['Text','Category'],
                      encoding='latin1')

prediction = bert_predictor.predict(test_data['Text'].to_list())
print(prediction)
