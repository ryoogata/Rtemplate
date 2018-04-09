# 共通ライブラリ
require(caret)
require(caretEnsemble)
require(doParallel)
require(dplyr)
require(magrittr)
require(myPackage)

# 固有ライブラリ
require(xgboost)

source("script/R/fun/tools.R", encoding = "utf8")
# source("script/R/fun/Data-pre-processing.R", encoding = "utf8")

# 保存した結果の読み込み
RESULT_DF <- "result.xgbLinear.df"

if ( file.exists(paste0("result/", RESULT_DF, ".data")) ) {
  assign(RESULT_DF, readRDS(paste0("result/", RESULT_DF, ".data")))
}


# データのファイルからの読み込み ----

# 学習用データの読み込み
STUDY <- data.table::fread("./data/div10_train.csv"
#STUDY <- data.table::fread("./data/train.csv"
                         ,stringsAsFactors = FALSE
                         ,sep = ","
                         ,data.table = FALSE
                         ,encoding = "UTF-8"
)

# 目的変数名を response に変更
names(STUDY)[names(STUDY) == "target"] <- "response"

# 目的変数 ( 0: no, 1: yes ) に変更
STUDY[STUDY$response == "yes", "response"] <- 1
STUDY[STUDY$response == "no", "response"] <- 0
STUDY$response <- as.numeric(STUDY$response)

# 前処理 ----

my_preProcess <- NULL
data_preProcess <- "dummy"

if ( data_preProcess == "none") {
  # 訓練用データの準備
  TRAIN.TRAIN <- dataPartition(STUDY) %>%
    magrittr::extract2(1)
  
  # テストデータの準備
  TRAIN.TEST <- dataPartition(STUDY) %>% 
    magrittr::extract2(2)
} else if ( data_preProcess == "dummy") {
  # 学習用データのダミー変数化
  STUDY %<>% myPackage::makeDummy()
  
  # 訓練用データの準備
  TRAIN.TRAIN <- dataPartition(STUDY) %>%
    magrittr::extract2(1)
  
  # テストデータの準備
  TRAIN.TEST <- dataPartition(STUDY) %>% 
    magrittr::extract2(2)
  
  # EVALUATION <- test.dummy
} else if ( data_preProcess == "nzv") {
  # STUDY <- all.nzv.train
  # TRAIN.TRAIN <- train.nzv.train
  # TRAIN.TEST <- train.nzv.test
  # EVALUATION <- test
} else if ( data_preProcess == "dummy.nzv.highlyCorDescr") {
  # STUDY <- train.dummy.nzv.highlyCorDescr
  # TRAIN.TRAIN <- train.dummy.nzv.highlyCorDescr.train
  # TRAIN.TEST <- train.dummy.nzv.highlyCorDescr.test
  # EVALUATION <- test.dummy.nzv.highlyCorDescr
}


#
# xgbLinear
#

# 変数指定 ( 共通設定 )
nresampling <- 10
n_repeats <- 10
METHOD <- "cv" # "repeatedcv", "boot"

# 変数指定 ( モデル個別 )
NROUNDS <- c(1)
LAMBDA <- c(.1)
ALPHA <- c(1)
ETA <- c(.1)

# seeds の決定
set.seed(123)
SEEDS <- vector(mode = "list", length = n_repeats * nresampling)
for(i in 1:(n_repeats * nresampling)) SEEDS[[i]] <- sample.int(1000, length(NROUNDS) * length(LAMBDA) * length(ALPHA) * length(ETA))
SEEDS[[n_repeats * nresampling + 1]] <- sample.int(1000, 1)

# 再抽出方法毎に INDEX を作成
set.seed(123)
if ( METHOD == "cv" ) {
  INDEX <- createFolds(TRAIN.TRAIN$response, k = nresampling, returnTrain = TRUE)
} else if ( METHOD == "repeatedcv" ){
  INDEX <- createMultiFolds(TRAIN.TRAIN$response, k = nresampling, times = n_repeats)
} else if ( METHOD == "boot" ){
  INDEX <- createResample(TRAIN.TRAIN$response, n_repeats)
}

set.seed(123)
doParallel <- trainControl(
  method = METHOD
  ,number = nresampling
  ,repeats = n_repeats
  ,classProbs = TRUE
  ,allowParallel = TRUE
  ,verboseIter = TRUE
  ,summaryFunction = mnLogLoss
  ,savePredictions = "final"
  ,index = INDEX
  ,seeds = SEEDS
)


# 説明変数一覧の作成
explanation_variable <- myPackage::exclude_variables(STUDY, c("response", "data_id"))

cl <- makeCluster(detectCores(), type = 'PSOCK', outfile = " ")
registerDoParallel(cl)

model_list <- caretList(
  x = TRAIN.TRAIN[,explanation_variable]
  ,y = TRAIN.TRAIN$response
  ,trControl = doParallel
  ,preProcess = my_preProcess
  ,tuneList = list(
    fit = caretModelSpec(
      method = "xgbLinear"
      ,metric = "logLoss" 
      ,label = TRAIN.TRAIN$response
      ,objective = "multi:softmax"
      ,num_class = 2
      ,tuneGrid = expand.grid(
        nrounds = NROUNDS
        ,lambda = LAMBDA
        ,alpha = ALPHA
        ,eta = ETA
      )
    )
  )
)

stopCluster(cl)
registerDoSEQ()

model_list[[1]]$times
model_list[[1]]$finalModel

#
# モデル比較
#
if (is.null(model_list[[1]]$preProcess)){
  # preProcess を指定していない場合
  TRAIN.SELECTED <- TRAIN.TEST 
} else {
  # preProcess を指定している場合
  TRAIN.SELECTED <- preProcess(
    subset(TRAIN.TEST, select = explanation_variable)
    ,method = my_preProcess
  ) %>%
    predict(., subset(TRAIN.TEST, select = explanation_variable))
}

# class を返す
# allPrediction <- caret::extractPrediction(
#                               list(model_list[[1]])
#                               ,testX = subset(TRAIN.SELECTED, select = explanation_variable)
#                               ,testY = unlist(subset(TRAIN.TEST, select = c(response)))
#                  )

# 確率を返す
allPrediction <- caret::extractProb(
                              list(model_list[[1]])
                              ,testX = subset(TRAIN.SELECTED, select = explanation_variable)
                              ,testY = unlist(subset(TRAIN.TEST, select = c(response)))
                 )

# dataType 列に Test と入っているもののみを抜き出す
testPrediction <- subset(allPrediction, dataType == "Test")
tp <- subset(testPrediction, object == "Object1")

# 精度確認 ( LogLoss )
MLmetrics::LogLoss(y_true = as.numeric(tp$obs), y_pred = as.numeric(tp$yes))

# 結果の保存
if (exists(RESULT_DF)){
  assign(RESULT_DF, dplyr::bind_rows(list(eval(parse(text = RESULT_DF)), summaryResult.LogLoss(model_list[[1]]))))
} else {
  assign(RESULT_DF, summaryResult.LogLoss(model_list[[1]]))
}
saveRDS(eval(parse(text = RESULT_DF)), paste0("result/", RESULT_DF, ".data"))

# predict() を利用した検算 
# pred_test.verification <- predict(model_list[[1]], TRAIN.SELECTED, type="raw") # class を返す
pred_test.verification <- predict(model_list[[1]], TRAIN.SELECTED, type="prob") # 確率を返す

# 精度確認 ( LogLoss )
MLmetrics::LogLoss(y_true = as.numeric(tp$obs), y_pred = as.numeric(pred_test.verification$yes))

# 学習用データ全てを利用してデルを作成
finalModel <- xgboost(
                      data = Matrix::Matrix(as.matrix(STUDY[,explanation_variable]), sparse=T)
                      ,label = STUDY$response
                      ,nrounds = 1
                      # General Parameters
                      ,booster = "gblinear"
                      # Booster Parameters for Linear Booster
                      ,params = list(lambda = .1
                                    ,alpha = .1
                                    ,eval_metric = "logloss"
                                )
                      # Task Parameters
                      ,objective = "reg:logistic"
                      )


# 評価用データの読み込み
EVALUATION <- data.table::fread("./data/test_updated.csv"
                         ,stringsAsFactors = FALSE
                         ,sep = ","
                         ,data.table = FALSE
                         ,encoding = "UTF-8"
)

# 評価用データのダミー変数化
EVALUATION %<>% myPackage::makeDummy()

#
# 評価用データにモデルの当てはめ
#
if (is.null(model_list[[1]]$preProcess)){
  # preProcess を指定していない場合
  pred_evaluation <- predict(object = finalModel, data = EVALUATION, type="response")[["predictions"]][,"yes"]
  pred_evaluation <- predict(finalModel
                             ,Matrix::Matrix(as.matrix(EVALUATION[,explanation_variable]), sparse=T)
                             )
  
  
  PREPROCESS <- "no_preProcess"
} else {
  # preProcess を指定している場合
  PREPROCESS <- preProcess(
      subset(EVALUATION, select = explanation_variable)
      ,method = my_preProcess
    ) %>%
      predict(., subset(EVALUATION, select = explanation_variable))
    
  pred_evaluation <- predict(object = finalModel, data = PREPROCESS, type="response")[["predictions"]][,"yes"]
  
  PREPROCESS <- paste(my_preProcess, collapse = "_")
}


#submitの形式で出力(CSV)
#データ加工
out <- data.frame(EVALUATION$data_id, pred_evaluation)

# 予測データを保存
for(NUM in 1:10){
  DATE <- format(jrvFinance::edate(from = Sys.Date(), 0), "%Y%m%d")
  SUBMIT_FILENAME <- paste("./submit/submit_", DATE, "_", NUM, "_", PREPROCESS, "_", model_list[[1]]$method, ".csv", sep = "")
  
  if ( !file.exists(SUBMIT_FILENAME) ) {
    write.table(out, #出力データ
                SUBMIT_FILENAME, #出力先
                quote = FALSE, #文字列を「"」で囲む有無
                col.names = FALSE, #変数名(列名)の有無
                row.names = FALSE, #行番号の有無
                sep = "," #区切り文字の指定
    )
    break
  }
}
