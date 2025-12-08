# 9.6 Deep Learning - Project: Electricity Prices Forecast

Antoine ANDRE, Arthur DESFRANCAIS, Lison SAUZAY

## 1. Context 

In this project, we try to predict electricity prices. 

Three models are implemented and evaluated.

For more details, see the report.

## 2. Structure

- Data: `240923 - suivi marché depuis 2012.csv`

- Models:
    1. ARIMA:
        - `arima.py`: performs the evaluation of the ARIMA model and compare withe Persistence model ;
    2. Chronos:
        - `chronos_eval.py`: provides methods to compute MASE and MDA ;
        - `chronos_zero_shot_prediction.py`: provides an example of zero-shot prediction, and plot the predicted 15 days, the expected 15 days, the previous 30 days ;
        - `chronos_zero_shot_eval.py`: performs the evaluation of zero-shot prediction and provides scores ;
        - `chronos_fine_tuned_train.py`: prepares data for fine-tuning (creates folder `chronos_fine_tuning_data`) and provides the command to execute in order to fine-tune the model ;
        - `chronos_fine_tuned_prediction.py`: provides an example of fine-tuned prediction, and plot the predicted 15 days, the expected 15 days, the previous 30 days ;
        - `chronos_fine_tuned_eval.py`: performs the evaluation of the fine-tuned model and provides scores ;
        - folder `chrono-forecasting`: Chronos scripts and model (imported from the Git repository made by Amazon).
    3. GRU:
        Available in the folder `GRU`.
        - folder `data`: dataset and data processing ;
        - folder `model`: run `predict.py` to obtain a visualization of predictions on the test set, other files useful to define and train the model.