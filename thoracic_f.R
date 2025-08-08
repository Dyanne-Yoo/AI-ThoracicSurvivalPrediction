
# 필요한 패키지 설치 및 로드
if (!requireNamespace("caret", quietly = TRUE)) install.packages("caret")
if (!requireNamespace("randomForest", quietly = TRUE)) install.packages("randomForest")
if (!requireNamespace("e1071", quietly = TRUE)) install.packages("e1071")
if (!requireNamespace("ROSE", quietly = TRUE)) install.packages("ROSE")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")

library(caret)
library(randomForest)
library(e1071)
library(ROSE)
library(ggplot2)
library(dplyr)

# 데이터 로드
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00277/ThoraricSurgery.arff"
data <- foreign::read.arff(url)


data$Risk1Yr <- as.factor(data$Risk1Yr)
data <- data %>%
  mutate(AGE = case_when(
    AGE >= 20 & AGE < 30 ~ "20-29",
    AGE >= 30 & AGE < 40 ~ "30-39",
    AGE >= 40 & AGE < 50 ~ "40-49",
    AGE >= 50 & AGE < 60 ~ "50-59",
    AGE >= 60 & AGE < 70 ~ "60-69",
    AGE >= 70 & AGE < 80 ~ "70-79",
    AGE >= 80 & AGE < 90 ~ "80-89",
    TRUE ~ as.character(AGE) 
  )) %>%
  mutate(AGE = factor(AGE, levels = c("20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89")))

# 이진 변수 처리
binary_vars <- c("PRE7", "PRE8", "PRE9", "PRE10", "PRE11",
                 "PRE17", "PRE19", "PRE25", "PRE30", "PRE32", "Risk1Yr")
data[binary_vars] <- lapply(data[binary_vars], function(x) ifelse(x == "T", 1, 0))

# PRE4와 PRE5 정규화
data$PRE4 <- round((data$PRE4 - min(data$PRE4)) / (max(data$PRE4) - min(data$PRE4)), 2)
data$PRE5 <- round((data$PRE5 - min(data$PRE5)) / (max(data$PRE5) - min(data$PRE5)), 2)

# PRE6과 PRE14 정렬 및 변환
data$PRE6 <- factor(data$PRE6, levels = c("PRZ0", "PRZ1", "PRZ2"), ordered = TRUE)
data$PRE6 <- as.numeric(data$PRE6) - 1

data$PRE14 <- factor(data$PRE14, levels = c("OC11", "OC12", "OC13", "OC14"), ordered = TRUE)
data$PRE14 <- as.numeric(data$PRE14) - 1

# 반응 변수 팩터화
data$Risk1Yr <- as.factor(data$Risk1Yr)

# 데이터 구조 확인
str(data)
summary(data)


#데이터 시각화 단계

# 필요한 패키지 로드
library(ggplot2)
library(RColorBrewer)

# DGN 코드 분포 시각화
ggplot(data, aes(x = DGN, fill = DGN)) +
  geom_bar() + 
  theme_minimal() +  
  scale_fill_brewer(palette = "Set2") +  
  labs(
    title = "Distribution of DGN Codes",
    x = "DGN Code",
    y = "Count"
  ) +
  theme(
    text = element_text(size = 14), 
    axis.text.x = element_text(angle = 45, hjust = 1) 
  )
  

# DGN별 1년 생존하지 못한 환자 시각화
data %>% 
  filter(Risk1Yr == 1) %>% 
  ggplot(aes(x = DGN, fill = DGN)) +
  geom_bar() +
  theme_minimal() +
  theme(text = element_text(size = 14)) +
  labs(
    title = "Distribution of DGN",
    x = "DGN Type",
    y = "Count"
  ) +
  scale_fill_brewer(palette = "Set3") 
  
  
#흡연자 수 빈도 시각화
ggplot(data, aes(x = factor(PRE30), fill = factor(PRE30))) +
  geom_bar() +
  theme_minimal() +
  scale_fill_brewer(palette = "Set2", name = "Smoker") +  
  labs(
    title = "Histogram of Smokers and Non-Smokers",
    x = "Smoker (0 = Non-Smoker, 1 = Smoker)",
    y = "Count"
  ) +
  theme(
    text = element_text(size = 14)
  )
  
  
#흡연여부별 생존,사망자 시각화 

data_grouped <- data %>%
  group_by(Risk1Yr, PRE30) %>%
  summarise(Count = n(), .groups = "drop")


ggplot(data_grouped, aes(x = interaction(Risk1Yr, PRE30), y = Count, fill = interaction(Risk1Yr, PRE30))) +
  geom_bar(stat = "identity") +
  scale_fill_brewer(palette = "Set2", name = "Risk & Smoker") + 
  labs(
    title = "Bar Chart of Risk by Smoking",
    x = "(Risk1Yr, Smoker)",
    y = "Count"
  ) +
  theme_minimal() +
  theme(
    text = element_text(size = 14),  
    axis.text.x = element_text(angle = 45, hjust = 1)  
  )
  
  
#PRE4 시각화 

library(ggplot2)

ggplot(data, aes(x = PRE4)) +
  geom_histogram(bins = 10, fill = "skyblue", color = "black") +
  labs(
    title = "Histogram of Forced Vital Capacity",
    x = "Exhaled Total Amount Of Air",
    y = "Number of patients"
  ) +
  theme_minimal() +
  theme(
    text = element_text(size = 14)  # 텍스트 크기 조정
  )
  

# PRE5 히스토그램
ggplot(data, aes(x = PRE5)) +
  geom_histogram(
    bins = 40, 
    fill = "skyblue", 
    color = "black", 
    alpha = 0.7
  ) +
  labs(
    title = "Histogram of PRE5",
    x = "PRE5 (Forced Expiratory Volume)",
    y = "Number of Patients"
  ) +
  theme_minimal()
  
  
# AGE 막대 그래프
ggplot(data, aes(x = AGE)) +
  geom_bar(fill = "lightcoral", color = "black", alpha = 0.7) +
  labs(
    title = "Age Group Distribution",
    x = "Age Group",
    y = "Count"
  ) +
  theme_minimal()
  
#종양크기별 개수 시각화 

ggplot(data, aes(x = PRE14, fill = PRE14)) +
  geom_bar() +
  scale_fill_brewer(palette = "Set3", name = "OC Type") +  # Set3 색상 팔레트 사용
  labs(
    title = "Type of OC in All Patients",
    x = "OC Type",
    y = "Count"
  ) +
  theme_minimal() +
  theme(
    text = element_text(size = 14),  # 텍스트 크기 조정
    axis.text.x = element_text(angle = 45, hjust = 1)  # x축 라벨 회전
  )

# 1년 생존하지 못한 환자 필터링
filtered_data <- data %>% filter(Risk1Yr == 1)

# 막대 그래프 생성
ggplot(filtered_data, aes(x = PRE14, fill = PRE14)) +
  geom_bar() +
  scale_fill_brewer(palette = "Set3", name = "OC Type") +  # Set3 색상 팔레트 사용
  labs(
    title = "Type of OC in Patients Who Didn't Survive First Year After Surgery",
    x = "OC Type",
    y = "Count"
  ) +
  theme_minimal() +
  theme(
    text = element_text(size = 14),  # 텍스트 크기 조정
    axis.text.x = element_text(angle = 45, hjust = 1)  # x축 라벨 회전
  )
  
  
library(tidyr)

# 이진 변수 목록
binary_vars <- c("PRE7", "PRE8", "PRE9", "PRE10", "PRE11", "PRE17", "PRE19", "PRE25", "PRE30", "PRE32")

# 이진 변수 시각화 반복
for (var in binary_vars) {
  print(
    ggplot(data, aes(x = as.factor(.data[[var]]), fill = as.factor(.data[[var]]))) +
      geom_bar() +
      labs(
        title = paste("Distribution of", var),
        x = var,
        y = "Count"
      ) +
      scale_fill_brewer(palette = "Set3", name = "Value") +  
      theme_minimal() +
      theme(
        text = element_text(size = 14),  # 텍스트 크기
        axis.text.x = element_text(angle = 45, hjust = 1)  
      )
  )
}
#상관계수 시각화
numeric_vars <- data %>% select_if(is.numeric)

cor_results <- cor(numeric_vars)

# 상관계수 시각화
library(corrplot)
corrplot::corrplot(cor_results, method = "color", type = "upper", tl.col = "black", tl.srt = 45)


  
    
  
# 생존 여부 막대 그래프
ggplot(data, aes(x = factor(Risk1Yr), fill = factor(Risk1Yr))) +
  geom_bar() +
  scale_fill_manual(values = c("darkred", "darkgreen")) +
  labs(title = "Survival After Surgery", x = "Survival (1 Year)", y = "Count") +
  theme_minimal()
  
  
  
  
  # ROSE로 데이터 불균형 해결
set.seed(42)
data <- ROSE::ovun.sample(
  Risk1Yr ~ .,
  data = data,
  method = "both",
  N = nrow(data) * 2
)$data


# glm 모델로 상관관계 도출
glm_model <- glm(Risk1Yr ~ ., data = data, family = binomial)
summary(glm_model)

significant_vars <- names(which(summary(glm_model)$coefficients[, 4] < 0.05))
significant_vars <- significant_vars[significant_vars != "(Intercept)"]

print(significant_vars)

dgn_encoded <- model.matrix(~ DGN - 1, data = data)
age_encoded <- model.matrix(~ AGE - 1, data = data)
numeric_data <- data[, sapply(data, is.numeric)]
factor_data <- data[, "Risk1Yr", drop = FALSE]

final_data <- cbind(numeric_data, dgn_encoded, age_encoded, factor_data)
colnames(final_data) <- gsub("-", "_", colnames(final_data))

selected_vars <- c("DGNDGN3", "DGNDGN5", "AGE40_49", "AGE50_59", "AGE60_69", "AGE70_79",
                   "PRE4", "PRE5", "PRE7", "PRE8", "PRE9", "PRE10", "PRE11", "PRE14", "PRE17", "PRE30", "Risk1Yr")

final_data_selected <- final_data[, selected_vars, drop = FALSE]
str(final_data_selected)



# 필요한 라이브러리 로드
library(caret)
library(dplyr)

# 데이터 분할 (70% 학습 데이터, 30% 테스트 데이터)
set.seed(42)
train_index <- createDataPartition(final_data_selected$Risk1Yr, p = 0.7, list = FALSE)
train_data <- final_data_selected[train_index, ]
test_data <- final_data_selected[-train_index, ]

# 랜덤 포레스트 모델 학습
rf_model <- randomForest(Risk1Yr ~ ., data = train_data, ntree = 100)
rf_predictions <- predict(rf_model, test_data)
rf_conf_matrix <- confusionMatrix(rf_predictions, test_data$Risk1Yr)
print(rf_conf_matrix)

# SVM 모델 학습
svm_model <- svm(Risk1Yr ~ ., data = train_data, kernel = "radial")
svm_predictions <- predict(svm_model, test_data)
svm_conf_matrix <- confusionMatrix(svm_predictions, test_data$Risk1Yr)
print(svm_conf_matrix)

# 선형 회귀 모델 학습
# 필요한 라이브러리 로드
library(pROC)
library(caret)

# 1. 선형 회귀 모델 학습 (GLM)
lm_model <- glm(Risk1Yr ~ ., data = train_data, family = binomial)

lm_predictions <- predict(lm_model, test_data, type = "response")

roc_curve <- roc(test_data$Risk1Yr, lm_predictions)

optimal_threshold <- coords(roc_curve, "best", ret = "threshold")$threshold

print(paste("Optimal Threshold: ", optimal_threshold))

lm_pred_class <- ifelse(lm_predictions > optimal_threshold, 1, 0)

test_data$Risk1Yr <- factor(test_data$Risk1Yr, levels = c(0, 1))
lm_pred_class <- factor(lm_pred_class, levels = c(0, 1))

lm_conf_matrix <- confusionMatrix(lm_pred_class, test_data$Risk1Yr)
print(lm_conf_matrix)


# SVM 하이퍼파라미터 튜닝
set.seed(42)

svm_grid <- expand.grid(
  C = 2^(-2:2),       
  sigma = 2^(-2:2)    
)

svm_control <- trainControl(
  method = "cv",  
  number = 5,     
  verboseIter = TRUE
)

svm_tuned <- train(
  Risk1Yr ~ .,
  data = train_data,
  method = "svmRadial",
  tuneGrid = svm_grid,
  trControl = svm_control
)

# 최적의 하이퍼파라미터 출력
print(svm_tuned$bestTune)

# 최적 모델 예측 및 평가
svm_best_predictions <- predict(svm_tuned, test_data)
svm_best_conf_matrix <- confusionMatrix(svm_best_predictions, test_data$Risk1Yr)
print(svm_best_conf_matrix)

# XGBoost 모델 학습
library(xgboost)
library(caret)


train_x <- as.matrix(train_data[, -ncol(train_data)])  
train_y <- as.numeric(train_data$Risk1Yr) - 1         
test_x <- as.matrix(test_data[, -ncol(test_data)])
test_y <- as.numeric(test_data$Risk1Yr) - 1


dtrain <- xgb.DMatrix(data = train_x, label = train_y)
dtest <- xgb.DMatrix(data = test_x, label = test_y)


params <- list(
  objective = "binary:logistic",  
  eval_metric = "logloss",       
  max_depth = 6,                 
  eta = 0.3,                     
  nthread = 2
  )

set.seed(42)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,       
  watchlist = list(train = dtrain, test = dtest),  
  verbose = 1
)

xgb_predictions <- predict(xgb_model, dtest)
xgb_pred_class <- ifelse(xgb_predictions > 0.5, 1, 0)


conf_matrix <- confusionMatrix(as.factor(xgb_pred_class), as.factor(test_y))
print(conf_matrix)