import xgboost as xgb
# xgbr = xgb.XGBRegressor(colsample_bytree=0.9,gamma=0.01, learning_rate= 0.1, max_depth= 9, min_child_weight=1, n_estimators= 500, reg_alpha= 0, reg_lambda= 1.5, subsample= 0.8)
# xgbr.fit(x_train, y_train)
# y_pred_xgbr = xgbr.predict(x_test)

# # Evaluation
# mae_xgbr = round(mean_absolute_error(y_test, y_pred_xgbr), 3)
# mse_xgbr = round(mean_squared_error(y_test, y_pred_xgbr), 3)
# rmse_xgbr = round(np.sqrt(mse_xgbr), 3)
# r2_value_xgbr = round(r2_score(y_test, y_pred_xgbr), 3)

# print('XGBoost Regressor Performance:')
# print('Mean Absolute Error:', mae_xgbr)
# print('Mean Squared Error:', mse_xgbr)
# print('Root Mean Squared Error:', rmse_xgbr)
# print('R-squared value:', r2_value_xgbr)
# # ### Note