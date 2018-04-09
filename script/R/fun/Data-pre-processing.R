require(caret)
require(dplyr)

# 学習用データ: モデルの学習・訓練に使用する全てのデータ ( study )
# 評価用データ: 実際に予測をするためのデータで kaggle などに提出するデータ ( evaluation )
# 
# 以下のデータは学習用データから作成
# 
# 訓練用データ: モデルを学習させるために利用するデータ ( train )
# 検査用データ: クロスバリデーション内でモデルの評価をするために利用するデータ ( validate )
# テストデータ: 完成したモデルを最終的に評価するためのデータ ( test ) 

# データの読み込み
STUDY <- data.table::fread("./data/train.csv"
                         ,stringsAsFactors = FALSE
                         ,sep = ","
                         ,data.table = FALSE
                         ,encoding = "UTF-8"
)

EVALUATION <- data.table::fread("./data/test.csv"
                         ,stringsAsFactors = FALSE
                         ,sep = ","
                         ,data.table = FALSE
                         ,encoding = "UTF-8"
)

# 目的変数名を response に変更
names(STUDY)[names(STUDY) == "y"] <- "response"