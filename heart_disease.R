# =============================================================
# 머신러닝 모델을 이용한 심장병 여부 예측 (R)
# 데이터: Kaggle Heart Failure Prediction (heart.csv, 918행 x 12열)
#
# 실제 분석 코드(노션 기록의 코드 캡처)를 그대로 옮겨 적은 스크립트입니다.
# 원본과 다른 부분은 [수정] 주석으로 표시했습니다.
#  - 평가 시 positive = "1"(심장병) 지정 → 민감도가 '환자를 찾아낸 비율'이 되도록
# =============================================================

# install.packages(c("caret", "randomForest", "kernlab", "ggplot2", "reshape2"))
library(caret)
library(randomForest)
library(kernlab)
library(ggplot2)
library(reshape2)

# ---------- 1. 데이터 불러오기 ----------
df <- read.csv("data/heart.csv")
head(df)
str(df)

# ---------- 2. 범주형 변수 factor 변환 ----------
continuous_features <- c("Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak")

# 연속형을 제외한 모든 변수 = 범주형 (타깃 포함)
features_to_convert <- setdiff(names(df), continuous_features)
df[features_to_convert] <- lapply(df[features_to_convert], as.factor)
str(df)

# ---------- 3. 시각화 ----------
# 연속형: 히스토그램
summary(df[continuous_features])
par(mfrow = c(2, 3))
for (feature in continuous_features) {
  hist(df[[feature]],
       main = paste("Distribution of", feature),
       xlab = feature,
       col = "lightblue",
       border = "grey",
       breaks = 30)
}
par(mfrow = c(1, 1))

# 범주형: 파이 차트
summary(df[features_to_convert])
for (feature in features_to_convert) {
  counts <- table(df[[feature]])
  percentages <- round(prop.table(counts) * 100, 1)
  pie(percentages,
      labels = paste0(names(percentages), " (", percentages, "%)"),
      main = paste("Pie Chart of", feature),
      col = rainbow(length(percentages)))
}
par(mfrow = c(1, 1))

# ---------- 4. 전처리 ----------
# 4-1. 0으로 기록된 콜레스테롤(172건)·안정 시 혈압(1건)은 결측으로 보고 중앙값 대체
sum(df$Cholesterol == 0)   # 172
sum(df$RestingBP == 0)     # 1

median_chol <- median(df$Cholesterol[df$Cholesterol != 0])  # 237
median_bp   <- median(df$RestingBP[df$RestingBP != 0])      # 130
df$Cholesterol[df$Cholesterol == 0] <- median_chol
df$RestingBP[df$RestingBP == 0]     <- median_bp
sum(df$Cholesterol == 0)   # 0
sum(df$RestingBP == 0)     # 0

# 4-2. 범주형 원-핫 인코딩 (타깃 제외)
categorical_features <- setdiff(features_to_convert, "HeartDisease")
dummy <- dummyVars(~ ., data = df[categorical_features])
cart_dummy <- data.frame(predict(dummy, df[categorical_features]))
cart_dummy <- cbind(cart_dummy, df["HeartDisease"])

# 4-3. 연속형 표준화 (Z-score)
cont <- c("Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak")
cont <- scale(df[, cont])
head(cont)

# 4-4. train / test 분리 (계층 샘플링 8:2)
final_df <- cbind(cart_dummy, cont)
x <- final_df[, setdiff(names(final_df), "HeartDisease")]
y <- final_df$HeartDisease

set.seed(26)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
x_train <- x[train_index, ]
x_test  <- x[-train_index, ]
y_train <- y[train_index]
y_test  <- y[-train_index]

# ---------- 5. 모델링 ----------
# [수정] 원본은 confusionMatrix()의 기본값(첫 번째 레벨 "0"이 positive)을 사용해
#        Sensitivity가 '정상인을 정상으로 맞힌 비율'로 출력되었습니다.
evaluate <- function(pred, truth) {
  confusionMatrix(factor(pred, levels = c(0, 1)),
                  factor(truth, levels = c(0, 1)),
                  positive = "1")
}

# 5-1. 랜덤 포레스트
set.seed(18)
model <- randomForest(x = x_train, y = as.factor(y_train), ntree = 500)
print(model)                 # OOB error 약 13.47%
y_pred <- predict(model, x_test)
conf_matrix <- evaluate(y_pred, y_test)
print(conf_matrix)
varImpPlot(model)

# 혼동행렬 히트맵
cm_df <- as.data.frame(conf_matrix$table)
cm_df$Prediction <- factor(cm_df$Prediction, levels = c(1, 0))
colnames(cm_df) <- c("Prediction", "Reference", "Freq")
ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1.2, color = "black", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "red") +
  theme_minimal() +
  labs(title = "실제값 VS 예측값 히트맵", x = "Reference", y = "Prediction")

# 5-2. SVM (선형 커널)
set.seed(26)
svm_trainX <- as.matrix(x_train)
svm_trainY <- as.factor(y_train)
svm_testX  <- as.matrix(x_test)

svm_model <- ksvm(x = svm_trainX, y = svm_trainY,
                  type = "C-svc", kernel = "vanilladot", C = 1)
print(svm_model)
conf_matrix_svm <- evaluate(predict(svm_model, svm_testX), y_test)
conf_matrix_svm

# 5-3. SVM (RBF 커널)
svm_model2 <- ksvm(x = svm_trainX, y = svm_trainY,
                   type = "C-svc", kernel = rbfdot(sigma = 0.1))
print(svm_model2)
conf_matrix_svm2 <- evaluate(predict(svm_model2, svm_testX), y_test)
conf_matrix_svm2

# 5-4. 로지스틱 회귀
logic_train <- data.frame(y = y_train, x_train)
logic_test  <- data.frame(y = y_test,  x_test)
result.glm  <- glm(y ~ ., data = logic_train, family = "binomial")
summary(result.glm)

sigmoid <- function(x) { 1 / (1 + exp(-x)) }
logic_prob <- sigmoid(predict(result.glm, newdata = logic_test))
logic_pred <- ifelse(logic_prob >= 0.5, 1, 0)
conf_matrix_logi <- evaluate(logic_pred, y_test)
conf_matrix_logi

# ---------- 6. 새 환자 예측 예시 (랜덤 포레스트) ----------
new_data <- x_test[1, ]   # 원본 캡처에 new_data를 만든 줄이 없어 테스트 1행을 바탕으로 사용
new_data$Age       <- (50  - mean(df$Age))       / sd(df$Age)
new_data$Sex.M     <- 1
new_data$Sex.F     <- 0
new_data$RestingBP <- (150 - mean(df$RestingBP)) / sd(df$RestingBP)
new_pred <- predict(model, new_data)
ifelse(new_pred == 1, "심장병 있음!", "심장병 없음!")
