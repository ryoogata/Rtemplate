summaryResult.Rsquared <- function(MODEL_LIST) {
  # control
  if ( is.null(MODEL_LIST$call$control)) {
    cont <- "NA"
  } else cont <- data.frame(control = MODEL_LIST$call$control)
  
  # preProcess
  if ( is.null(names(MODEL_LIST$preProcess$method))){
    prep <- "NA"
  } else prep <- names(MODEL_LIST$preProcess$method)
  
  # tuneGrid
  if ( is.null(MODEL_LIST$call$tuneGrid)){
    grid <- "NA"
  } else {
    grid <- apply(MODEL_LIST$call$tuneGrid, 2, paste, collapse = ",") %>%
    stringr::str_split(string = ., pattern = ",") %>%
    lapply(., unique) %>%
    lapply(., paste, collapse = ",") %>%
    data.frame(.)
  }
  
  names(grid) <- paste0("tuneGrid.", names(model_list[[1]]$call$tuneGrid))
  
  # 
  df <- data.frame(
    contest_Rsquared = 0
    ,test_Rsquared = rsquaredSummary(tp)
    ,train_Rsquared = max(model_list[[1]]$results[,model_list[[1]]$metric])
    ,method = MODEL_LIST$method # モデル作成アルゴリズム
    ,elapsed = MODEL_LIST$times$everything["elapsed"] # 実行時間
    ,data_preProcess = data_preProcess
    ,caret_preProcess = data.frame(caret_preProcess = paste(prep, collapse = ", "))
    ,length_exp = length(explanation_variable) # 説明変数の数
    #,exp = data.frame(exp = paste(explanation_variable, collapse = ", ")) # 説明変数の値
    ,bestTune = MODEL_LIST$bestTune # 最適なパラメータの値
    ,grid
    ,cont
    ,trainControl_method = model_list[[1]]$control$method
    ,number = model_list[[1]]$control$number
    ,repeats = model_list[[1]]$control$repeats
    ,p = model_list[[1]]$control$p
    ,row.names = NULL
    ,stringsAsFactors = FALSE
  )
  
  return(df)
}

summaryResult.ROC <- function(MODEL_LIST) {
  # control
  if ( is.null(MODEL_LIST$call$control)) {
    cont <- "NA"
  } else cont <- data.frame(control = MODEL_LIST$call$control)
  
  # preProcess
  if ( is.null(names(MODEL_LIST$preProcess$method))){
    prep <- "NA"
  } else prep <- names(MODEL_LIST$preProcess$method)
  
  # tuneGrid
  if ( is.null(MODEL_LIST$call$tuneGrid)){
    grid <- "NA"
  } else grid <- apply(MODEL_LIST$call$tuneGrid, 2, paste, collapse = ",") %>%
    stringr::str_split(string = ., pattern = ",") %>%
    lapply(., unique) %>%
    lapply(., paste, collapse = ",") %>%
    data.frame(.)
  
  
  names(grid) <- paste0("tuneGrid.", names(model_list[[1]]$call$tuneGrid))
  
  # 
  df <- data.frame(
    test_ROC = pROC::roc(tp$obs, tp$yes)$auc[1]
    ,train_ROC = max(model_list[[1]]$results[,model_list[[1]]$metric])
    ,method = MODEL_LIST$method
    ,elapsed = MODEL_LIST$times$everything["elapsed"]
    ,data_preProcess = data_preProcess
    ,caret_preProcess = data.frame(caret_preProcess = paste(prep, collapse = ", "))
    ,length_exp = length(explanation_variable)
    ,exp = data.frame(exp = paste(explanation_variable, collapse = ", "))
    ,bestTune = MODEL_LIST$bestTune
    ,grid
    ,cont
    ,row.names = NULL
    ,stringsAsFactors = FALSE
  )
  
  return(df)
}


summaryResult.RMSE <- function(MODEL_LIST) {
  # control
  if ( is.null(MODEL_LIST$call$control)) {
    cont <- "NA"
  } else cont <- data.frame(control = MODEL_LIST$call$control)
  
  # preProcess
  if ( is.null(names(MODEL_LIST$preProcess$method))){
    prep <- "NA"
  } else prep <- names(MODEL_LIST$preProcess$method)
  
  # tuneGrid
  if ( is.null(MODEL_LIST$call$tuneGrid)){
    grid <- "NA"
  } else grid <- apply(MODEL_LIST$call$tuneGrid, 2, paste, collapse = ",") %>%
    stringr::str_split(string = ., pattern = ",") %>%
    lapply(., unique) %>%
    lapply(., paste, collapse = ",") %>%
    data.frame(.)
  
  
  names(grid) <- paste0("tuneGrid.", names(model_list[[1]]$call$tuneGrid))
  
  # 
  df <- data.frame(
    contest_RMSE = 0
    ,test_RMSE = sqrt(sum((tp$obs - tp$pred)^2)/nrow(tp))
    ,train_RMSE = min(model_list[[1]]$results[,model_list[[1]]$metric])
    ,method = MODEL_LIST$method
    ,elapsed = MODEL_LIST$times$everything["elapsed"]
    ,data_preProcess = data_preProcess
    ,caret_preProcess = data.frame(caret_preProcess = paste(prep, collapse = ", "))
    ,length_exp = length(explanation_variable)
    ,exp = data.frame(exp = paste(explanation_variable, collapse = ", "))
    ,bestTune = MODEL_LIST$bestTune
    ,grid
    ,cont
    ,row.names = NULL
    ,stringsAsFactors = FALSE
  )
  
  return(df)
}


summaryResult.LogLoss <- function(MODEL_LIST) {
  # control
  if ( is.null(MODEL_LIST$call$control)) {
    cont <- "NA"
  } else cont <- data.frame(control = MODEL_LIST$call$control)
  
  # preProcess
  if ( is.null(names(MODEL_LIST$preProcess$method))){
    prep <- "NA"
  } else prep <- names(MODEL_LIST$preProcess$method)
  
  # tuneGrid
  if ( is.null(MODEL_LIST$call$tuneGrid)){
    grid <- "NA"
  } else grid <- apply(MODEL_LIST$call$tuneGrid, 2, paste, collapse = ",") %>%
    stringr::str_split(string = ., pattern = ",") %>%
    lapply(., unique) %>%
    lapply(., paste, collapse = ",") %>%
    data.frame(.)
  
  
  names(grid) <- paste0("tuneGrid.", names(model_list[[1]]$call$tuneGrid))
  
  # 
  df <- data.frame(
    test_LogLoss = MLmetrics::MultiLogLoss(y_true = tp$obs, y_pred = y_pred)
    ,train_LogLoss = min(model_list[[1]]$results[,model_list[[1]]$metric])
    ,method = MODEL_LIST$method
    ,elapsed = MODEL_LIST$times$everything["elapsed"]
    ,data_preProcess = data_preProcess
    ,caret_preProcess = data.frame(caret_preProcess = paste(prep, collapse = ", "))
    ,length_exp = length(explanation_variable)
    ,exp = data.frame(exp = paste(explanation_variable, collapse = ", "))
    ,bestTune = MODEL_LIST$bestTune
    ,grid
    ,cont
    ,row.names = NULL
    ,stringsAsFactors = FALSE
  )
  
  return(df)
}
