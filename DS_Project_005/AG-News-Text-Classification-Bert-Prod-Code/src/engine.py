from pathlib import Path
import sys
import warnings
warnings.simplefilter(action='ignore')

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))

from src.ML_Pipeline import model
from src.ML_Pipeline import utils
from src.ML_Pipeline import feature_engineering

MAX_LEN=512


try:
    # Load Dataset and show details:
    print('##### Load Dataset and Show Details #####')
    utils.load_and_display_dataset_details()

    # Load Train-Test data and convert to DataFrame Object for further operations:
    print('##### Load Train-Test data and convert to DataFrame Object for further operations #####')
    ag_news_train_df, ag_news_test_df, class_label_names = utils.load_and_convert_data_to_df()

    # Data Preprocessing using K-Train:
    print('##### Data Preprocessing using K-Train #####')
    (X_train, y_train), (X_test, y_test), preprocessing_var = \
        feature_engineering.perform_feature_engineering(ag_news_train_df, ag_news_test_df, MAX_LEN)

    # Create & Train BERT Model:
    print('##### Create & Train BERT Model #####')
    bert_learner = \
        model.create_and_train_bert_model(X_train, y_train, X_test, y_test, preprocessing_var)

    # Check Model performance during training and validation:
    print('##### Check Model performance during training and validation #####')
    model.check_model_performance(bert_learner, class_label_names)

    # Saving Bert Model Fine-tuned on AG News Dataset:
    print('##### Saving Bert Model Fine-tuned on AG News Dataset #####')
    model.save_fine_tuned_bert_model(bert_learner, preprocessing_var)

except Exception as e:
    print('!! Exception Details: !!\n', e.__class__)
    print('Please debug for further details')
