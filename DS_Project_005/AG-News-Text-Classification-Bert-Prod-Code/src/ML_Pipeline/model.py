import ktrain
import timeit
from ktrain import text

NUM_EPOCHS=3
LEARNING_RATE=2e-5
BATCH_SIZE=6
MIN_SECONDS=60
MAX_LEN=512


# function to create and train BERT model
def create_and_train_bert_model(X_train, y_train, X_test, y_test, preproc_var):
    transformer_bert_model = text.text_classifier(name='bert',
                                                  train_data=(X_train, y_train),
                                                  preproc=preproc_var)
    print("Transformer Layers: \n", transformer_bert_model.layers)
    print(f"\nCompiling & Training BERT for maxlen={MAX_LEN} & batch_size={BATCH_SIZE}")
    bert_learner = ktrain.get_learner(model=transformer_bert_model,
                                      train_data=(X_train, y_train),
                                      val_data=(X_test, y_test),
                                      batch_size=BATCH_SIZE)

    start_time = timeit.default_timer()
    print(f"\nFine Tuning BERT on AG News Dataset with \
learning rate={LEARNING_RATE} and epochs={NUM_EPOCHS}")
    bert_learner.fit_onecycle(lr=LEARNING_RATE, epochs=NUM_EPOCHS)
    stop_time = timeit.default_timer()
    print("Total training time in minutes: \n", (stop_time - start_time) / MIN_SECONDS)
    return bert_learner

# evaluate the performance of the model
def check_model_performance(bert_learner, class_label_names):
    print("BERT Performance Metrics on AG News Dataset :\n", bert_learner.validate())
    print("BERT Performance Metrics on AG News Dataset with Class Names :\n",
          bert_learner.validate(class_names=class_label_names))
    return None

# save the model
def save_fine_tuned_bert_model(bert_learner, preproc_var):
    bert_predictor = ktrain.get_predictor(bert_learner.model, preproc=preproc_var)
    bert_predictor.save('AG-News-Text-Classification-Bert-Prod-Code/output/bert-ag-news-predictor')
    return None

# reload the model
def load_model():
    bert_predictor = \
        ktrain.load_predictor('AG-News-Text-Classification-Bert-Prod-Code/output/bert-ag-news-predictor')
    print("Bert model loaded successfully: \n", bert_predictor.get_classes())
    return bert_predictor
