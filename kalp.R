library(ggplot2)
library(ggthemes)
library(plotly)
library(dplyr)
library(psych)
library(corrplot)
library(class)
library(tidyverse)
library(caret)
library(zoo)
library(e1071)
library(rstatix)
library(rpart)
library(neuralnet)
library(randomForest)


kalp <- read.csv("C:/Users/.../Desktop/kalp.csv")
dim(kalp)
head(kalp)
tail(kalp)
summary(kalp)

str(kalp)
nrow(kalp)
ncol(kalp)
colnames(kalp)

nullVarMı <- sum(is.na(kalp))
cat(nullVarMı)
head(kalp)
table(kalp$age)
table(kalp$sex)

age=table(kalp$age)
sex=table(kalp$sex)
cp= table(kalp$cp)
trestbps <- table(kalp$trestbps)
chol <- table(kalp$chol)
fbs <- table(kalp$fbs)
restecg <- table(kalp$restecg)
thalach <- table(kalp$thalach)
exang <- table(kalp$exang)
oldpeak <- table(kalp$oldpeak)
slope <- table(kalp$slope)
ca <- table(kalp$ca)
thal <- table(kalp$thal)


ggplot(kalp, aes(x = age)) + 
  geom_histogram(binwidth = 2, fill = "skyblue", color = "black") + 
  labs(title = "Yaş Dağılımı")
table(kalp$age)


barplot(sex , names.arg = c("Kadın" , "Erkek" ) ,
        col = c("pink","blue"))
table(kalp$sex)


pie(cp ,labels = c ("Tip-0","Tip-1","Tip-2","Tip-3"),
    col = c("orange","green","lightblue","purple"))
table(kalp$cp)
percent <- round(prop.table(cp) * 100,digits = 2)   
legend("topright", legend = paste0(names(cp), " ", percent, "%"))
table(kalp$cp)


ggplot(kalp, aes(x = trestbps)) + 
  geom_histogram(binwidth = 5, fill = "lightgreen", color = "black") + 
  labs(title = "Dinlenme Kan Basıncı Dağılımı")
table(kalp$trestbps)


ggplot(kalp, aes(x = chol)) + 
  geom_histogram(binwidth = 5, fill = "skyblue", color = "black") + 
  labs(title = "Kolesterol Dağılımı")
table(kalp$chol)


barplot(fbs , names.arg = c("Kan Şekeri Düşük" , "Kan Şekeri Yüksek" ) ,
        col = c("red","orange"))
table(kalp$fbs)


pie(restecg ,labels = c ("Sağlıklı","Düşük Risk","Riskli"),
    col = c("orange","green","purple"))
percent <- round(prop.table(restecg) * 100,digits = 2)    
legend("topright", legend = paste0(names(restecg), " ", percent, "%"))
table(kalp$restecg)


ggplot(kalp, aes(x = thalach)) + 
  geom_histogram(binwidth = 5, fill = "salmon", color = "black") +
  labs(title = "Maksimum Kalp Atış Hızı Dağılımı")


barplot(exang , names.arg = c("Yok" , "Var" ) ,
        col = c("magenta","gray"),
        main = "Egzersize Bağlı Angina",
        xlab = "Egzersize Bağlı Angina Durumu",
        ylab = "Frekans")
table(kalp$exang)


plot(oldpeak, type = "o", col = "red", pch = 16, lwd = 2, xlab = "Sıralama", ylab = "Değerler", main = "Değerlerin Sıralı Grafiği")
table(kalp$oldpeak)


pie(slope ,labels = c ("0","1","2"),
    col = c("orange","green","purple"))
table(kalp$slope)
percent <- round(prop.table(slope) * 100,digits = 2)    
legend("topright", legend = paste0(names(slope), " ", percent, "%"))


pie(thal ,labels = c ("0","Normal","Sabit kusur","Tersine çevrilebilir kusur"),
    col = c("orange","green","lightblue","purple"),)
table(kalp$thal)
percent <- round(prop.table(thal) * 100,digits = 2)    
legend("topright", legend = paste0(names(thal), " ", percent, "%"))


barplot(ca,col = "blue",ylim = c(0,200))
table(kalp$ca)

kalp_yeni <- kalp
kalp_yeni[sample(1:nrow(kalp),10),"age"] <- NA
print(sum(is.na(kalp_yeni)))


en_cok_tekrar_edilen <- names(age)[which.max(age)]

print(paste("En çok tekrar eden değer:", en_cok_tekrar_edilen))
print(paste("Tekrar sayısı:", age[en_cok_tekrar_edilen]))
print(kalp_yeni$age)

kalp_yeni$age <- 
  ifelse(is.na(kalp_yeni$age),
         en_cok_tekrar_edilen,
         kalp_yeni$age)

kalp_yeni$age <- as.integer(kalp_yeni$age)

table(kalp_yeni$age)

print(kalp_yeni)


# Min-max normalizasyonu
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Tüm özellikleri ölçeklendirme
kalp_yeni[,1:13] <- lapply(kalp_yeni[,1:13], normalize)

# Normalized veri
print(kalp_yeni)


# Z-Skor normalizasyonu (standartlaştırma)
standardize <- function(x) {
  return ((x - mean(x)) / sd(x))
}

kalp_yeni[,1:13] <- lapply(kalp_yeni[,1:13], standardize)

print(kalp_yeni)

outliers <- identify_outliers_zscore(kalp_yeni)
print(outliers)


replace_outliers_with_max <- function(data, threshold = 3) {
  for (col in colnames(data)) {
    z_scores <- scale(data[[col]])
    outlier_indices <- which(abs(z_scores) > threshold)
    non_outlier_values <- data[[col]][-outlier_indices]
    max_non_outlier <- max(non_outlier_values)
    data[[col]][outlier_indices] <- max_non_outlier
  }
  return(data)
}

# Aykırı değerleri değiştir
kalp_yeni <- replace_outliers_with_max(kalp_yeni)

# Değiştirilmiş verileri göster
print(kalp_yeni)

ggplot(kalp_yeni, aes(x = chol)) + 
  geom_histogram(binwidth = 0.1, fill = "skyblue", color = "black") + 
  labs(title = "Kolesterol Dağılımı")
table(kalp_yeni$chol)

--- KNN ---

set.seed(123)
ind <- sample(3, nrow(kalp_yeni), replace = TRUE, prob = c(0.7, 0.15, 0.15))
training <- kalp_yeni[ind == 1,]
validating <- kalp_yeni[ind == 2,]
testing <- kalp_yeni[ind == 3,]

k_degerleri <- seq(1, 31, by = 2)
dogruluk_deger <- numeric(length(k_degerleri))

for (k in k_degerleri) {
  tahmin <- knn(train = training[, 1:13],
                test = validating[, 1:13],
                cl = training$target,
                k = k)
  dogruluk_deger[which(k_degerleri == k)] <- 
    sum(tahmin == validating$target) / length((validating$target))
}

dogruluk_df <- data.frame(k = k_degerleri, dogruluk = dogruluk_deger)

print(dogruluk_df)

# Doğruluk değerlerini görselleştirme
ggplot(dogruluk_df, aes(x = k, y = dogruluk)) +
  geom_line() +
  geom_point() +
  labs(title = "kNN Model Doğruluğu", x = "k Değeri", y = "Doğruluk")

enIyi_k <- k_degerleri[which.max(dogruluk_deger)]
print(enIyi_k)

# En iyi k değeri ile modeli test seti üzerinde eğitme
test_model <- knn(
  train = training[, 1:13],  
  test = testing[, 1:13],     
  cl = training$target,       
  k = enIyi_k                         
)

confusion_matrix <- table(Reference=validating$target,prediction=tahmin)

print(confusion_matrix)

doğruluk <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
doğruluk=doğruluk*100

confusion_matrix_test <- table(Reference = testing$target, prediction = test_model)
print(confusion_matrix_test)

accuracy_test <- sum(diag(confusion_matrix_test)) / sum(confusion_matrix_test)  # Doğruluk hesaplama
accuracy_test = accuracy_test * 100 

print(enIyi_k)
cat("Modelin doğruluğu:",doğruluk)
cat("Modelin doğruluğu (test seti üzerinde):", accuracy_test)


# Doğruluk (accuracy) hesaplama
accuracyKnn <- sum(diag(confusion_matrix_test)) / sum(confusion_matrix_test)

# Duyarlılık (sensitivity) hesaplama
sensitivityKnn <- confusion_matrix_test[2, 2] / sum(confusion_matrix_test[2, ])

# Özgüllük (specificity) hesaplama
specificityKnn <- confusion_matrix_test[1, 1] / sum(confusion_matrix_test[1, ])

# F1-Skor hesaplama
precisionKnn <- confusion_matrix_test[2, 2] / sum(confusion_matrix_test[, 2])
recallKnn <- sensitivityKnn
f1_scoreKnn <- 2 * ((precisionKnn * recallKnn) / (precisionKnn + recallKnn))

# Sonuçları yazdırma
cat("Doğruluk (Accuracy):", accuracyKnn, "\n")
cat("Duyarlılık (Sensitivity):", sensitivityKnn, "\n")
cat("Özgüllük (Specificity):", specificityKnn, "\n")
cat("F1-Skor:", f1_scoreKnn, "\n")

--- NN ---

set.seed(123)
ind <- sample(3, nrow(kalp_yeni), replace = TRUE, prob = c(0.7, 0.15,0.15))
trainnn <- kalp_yeni[ind == 1,]
validationnn <- kalp_yeni[ind == 2,]
testnn <- kalp_yeni[ind == 3,]


nn <- neuralnet(target ~ ., 
                data = trainnn, 
                hidden = c(5, 3, 2), 
                linear.output = FALSE)
nn$result.matrix

model_resultsv <- compute(nn, validationnn)
predictednnv <- model_resultsv$net.result

cor(predictednnv, validationnn$target)

nn.resultsv <- compute(nn, validationnn[, -14])
resultsv <- data.frame(actual = validationnn$target, predictionnn = nn.resultsv$net.result)


rounded_resultsv <- sapply(resultsv, round, digits = 0)
rounded_results_dfv <- data.frame(rounded_resultsv)

 
conf_matrix_nnv <- table(actual = rounded_results_dfv$actual, predictionnn = rounded_results_dfv$predictionnn)
print(conf_matrix_nnv)

accuracyv <- sum(diag(conf_matrix_nnv)) / sum(conf_matrix_nnv)
cat("Doğruluk (Accuracy)Validation:", accuracyv, "\n")

conf_matrix_nn <- table(actual = rounded_results_df$actual, predictionnn = rounded_results_df$predictionnn)
print(conf_matrix_nn)

accuracynn <- sum(diag(conf_matrix_nn)) / sum(conf_matrix_nn)
cat("Doğruluk (Accuracy):", accuracynn, "\n")



model_results <- compute(nn, testnn)
predictednn <- model_results$net.result

cor(predictednn, testnn$target)

nn.results <- compute(nn, testnn[, -14])
results <- data.frame(actual = testnn$target, predictionnn = nn.results$net.result)


rounded_results <- sapply(results, round, digits = 0)
rounded_results_df <- data.frame(rounded_results)


#  Accuracy, Sensitivity, Specificity, F1-Score
accuracynn <- sum(diag(conf_matrix_nn)) / sum(conf_matrix_nn)
sensitivity <- conf_matrix_nn[2, 2] / sum(conf_matrix_nn[2, ])
specificity <- conf_matrix_nn[1, 1] / sum(conf_matrix_nn[1, ])
precision <- conf_matrix_nn[2, 2] / sum(conf_matrix_nn[, 2])
recall <- sensitivity
f1_score <- 2 * ((precision * recall) / (precision + recall))


cat("Doğruluk (Accuracy):", accuracynn, "\n")
cat("Duyarlılık (Sensitivity):", sensitivity, "\n")
cat("Özgüllük (Specificity):", specificity, "\n")
cat("F1-Skor:", f1_score, "\n")


--- RF ---

set.seed(123)
ind <- sample(3, nrow(kalp_yeni), replace = TRUE, prob = c(0.7, 0.15, 0.15))
trainrf <- kalp_yeni[ind == 1,]
validationrf <- kalp_yeni[ind == 2,]
testrf <- kalp_yeni[ind == 3,]

kalp_rf <- randomForest(target ~ ., data=trainrf,ntree = 500)
print(kalp_rf)

rf.predictionv <- predict(kalp_rf, newdata = validationrf)
rf.predictionv <- ifelse(rf.predictionv >= 0.5, 1, 0)


conf_matrixv <- table(validationrf$target, rf.predictionv)
print(conf_matrixv)

accuracyv <- sum(diag(conf_matrixv)) / sum(conf_matrixv)
cat("Doğruluk (Accuracy):", accuracyv, "\n")

rf.prediction <- predict(kalp_rf, newdata = testrf)
rf.prediction <- ifelse(rf.prediction >= 0.5, 1, 0)


# Confusion matrix oluşturma
conf_matrix <- table(testrf$target, rf.prediction)
print(conf_matrix)
accuracyrf <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Doğruluk (Accuracy):", accuracyrf, "\n")

# Doğruluk (Accuracy)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

# Duyarlılık (Sensitivity)
sensitivity <- conf_matrix[2, 2] / sum(conf_matrix[2, ])

# Özgüllük (Specificity)
specificity <- conf_matrix[1, 1] / sum(conf_matrix[1, ])

# F1-Skor hesaplama
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- sensitivity
f1_score <- 2 * ((precision * recall) / (precision + recall))

cat("Doğruluk (Accuracy):", accuracy, "\n")
cat("Duyarlılık (Sensitivity):", sensitivity, "\n")
cat("Özgüllük (Specificity):", specificity, "\n")
cat("F1-Skor:", f1_score, "\n")


--- SVM ---


set.seed(123)
ind <- sample(3, nrow(kalp_yeni), replace = TRUE, prob = c(0.7, 0.15, 0.15))
trainsvm <- kalp_yeni[ind == 1,]
validationsvm <- kalp_yeni[ind == 2,]
testsvm <- kalp_yeni[ind == 3,]

svm.linear <- svm(target ~ ., data = trainsvm, kernel = "linear", cost = 0.01, gamma = 0.001)
svm.radial <- svm(target ~ ., data = trainsvm, kernel = "radial", cost = 1, gamma = 0.1)

svm.linear.predval <- predict(svm.linear, newdata = validationsvm)
svm.linear.predval <- ifelse(svm.linear.predval >= 0.5, 1, 0)
table(validationsvm$target, svm.linear.predval)
sum(validationsvm$target == svm.linear.predval) / nrow(validationsvm)

svm.radial.predval <- predict(svm.radial, newdata = validationsvm)
svm.radial.predval <- ifelse(svm.radial.predval >= 0.5, 1, 0)
table(validationsvm$target, svm.radial.predval)
sum(validationsvm$target == svm.radial.predval) / nrow(validationsvm)

svm.linear.pred <- predict(svm.linear, newdata = testsvm)
svm.linear.pred <- ifelse(svm.linear.pred >= 0.5, 1, 0)
table(testsvm$target, svm.linear.pred)
sum(testsvm$target == svm.linear.pred) / nrow(testsvm)

svm.radial.pred <- predict(svm.radial, newdata = testsvm)
svm.radial.pred <- ifelse(svm.radial.pred >= 0.5, 1, 0)
table(testsvm$target, svm.radial.pred)
sum(testsvm$target == svm.radial.pred) / nrow(testsvm)


# Doğruluk (Accuracy)
accuracy_linear <- sum(testsvm$target == svm.linear.pred) / nrow(testsvm)
accuracy_radial <- sum(testsvm$target == svm.radial.pred) / nrow(testsvm)

# Confusion Matrix for Linear SVM
conf_matrix_linear <- table(testsvm$target, svm.linear.pred)
TP_linear <- conf_matrix_linear[2, 2]
TN_linear <- conf_matrix_linear[1, 1]
FP_linear <- conf_matrix_linear[1, 2]
FN_linear <- conf_matrix_linear[2, 1]

# Confusion Matrix for Radial SVM
conf_matrix_radial <- table(testsvm$target, svm.radial.pred)
TP_radial <- conf_matrix_radial[2, 2]
TN_radial <- conf_matrix_radial[1, 1]
FP_radial <- conf_matrix_radial[1, 2]
FN_radial <- conf_matrix_radial[2, 1]

# Precision hesaplama
precision_linear <- TP_linear / (TP_linear + FP_linear)
precision_radial <- TP_radial / (TP_radial + FP_radial)

# Recall (Duyarlılık) hesaplama
recall_linear <- sensitivity_linear
recall_radial <- sensitivity_radial

# Duyarlılık (Sensitivity)
sensitivity_linear <- TP_linear / (TP_linear + FN_linear)
sensitivity_radial <- TP_radial / (TP_radial + FN_radial)

# Özgüllük (Specificity)
specificity_linear <- TN_linear / (TN_linear + FP_linear)
specificity_radial <- TN_radial / (TN_radial + FP_radial)

# F1-Skor
f1_score_linear <- 2 * ((precision_linear * recall_linear) / (precision_linear + recall_linear))
f1_score_radial <- 2 * ((precision_radial * recall_radial) / (precision_radial + recall_radial))

# Sonuçların Yazdırılması
cat("Linear SVM Sonuçları:\n")
cat("Doğruluk (Accuracy - Linear):", accuracy_linear, "\n")
cat("Duyarlılık (Sensitivity - Linear):", sensitivity_linear, "\n")
cat("Özgüllük (Specificity - Linear):", specificity_linear, "\n")
cat("F1-Skor (Linear):", f1_score_linear, "\n\n")

cat("Radial SVM Sonuçları:\n")
cat("Doğruluk (Accuracy - Radial):", accuracy_radial, "\n")
cat("Duyarlılık (Sensitivity - Radial):", sensitivity_radial, "\n")
cat("Özgüllük (Specificity - Radial):", specificity_radial, "\n")
cat("F1-Skor (Radial):", f1_score_radial, "\n")


--- GLM ---

set.seed(123)
ind <- sample(3, nrow(kalp_yeni), replace = TRUE, prob = c(0.7, 0.15, 0.15))
trainglm <- kalp_yeni[ind == 1,]
validationglm <- kalp_yeni[ind == 2,]
testglm <- kalp_yeni[ind == 3,]

# Modelin eğitimi
log.model <- glm(target ~ ., data = trainglm, family = binomial(link = 'logit'))
summary(log.model)

# Validation seti üzerinde tahminler ve değerlendirme
glm.predictionval <- predict(log.model, newdata = validationglm, type = "response")
glm.predictionval <- ifelse(glm.predictionval >= 0.5, 1, 0)


table(validationglm$target, glm.predictionval)
sum(validationglm$target==glm.predictionval) / nrow(validationglm)

# Test seti üzerinde tahminler ve değerlendirme
glm.prediction <- predict(log.model, newdata = testglm, type = "response")
glm.prediction <- ifelse(glm.prediction >= 0.5, 1, 0)

table(testglm$target, glm.prediction)
sum(testglm$target==glm.prediction) / nrow(testglm)

conf_matrix_test <- table(testglm$target, glm.prediction)
accuracy_test <- sum(diag(conf_matrix_test)) / sum(conf_matrix_test)
sensitivity_test <- conf_matrix_test[2, 2] / sum(conf_matrix_test[2, ])
specificity_test <- conf_matrix_test[1, 1] / sum(conf_matrix_test[1, ])
precision_test <- conf_matrix_test[2, 2] / sum(conf_matrix_test[, 2])
recall_test <- sensitivity_test
f1_score_test <- 2 * ((precision_test * recall_test) / (precision_test + recall_test))


cat("Test Seti Doğruluk (Accuracy):", accuracy_test, "\n")
cat("Test Seti Duyarlılık (Sensitivity):", sensitivity_test, "\n")
cat("Test Seti Özgüllük (Specificity):", specificity_test, "\n")
cat("Test Seti F1-Skor:", f1_score_test, "\n")

---------------

accuracy_values <- c(accuracy_test, accuracynn, accuracyrf, accuracy_radial, accuracy_test)*100
models <- c("KNN", "NN", "RF", "SVM", "GLM")

accuracy_df <- data.frame(Model = models, Accuracy = accuracy_values)


ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Modellerin Doğruluk Değerleri", x = "Model", y = "Doğruluk") +
  ylim(0,100)+
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Model isimleri ve doğruluk değerleri
model_names <- c("KNN", "NN", "RF", "SVM", "GLM")
accuracy_scores <- c(accuracy_test, accuracynn, accuracyrf, accuracy_linear, accuracy_test)

# Doğruluk değerleri ve modelleri bir veri çerçevesinde birleştirme
accuracy_df <- data.frame(Model = model_names, Accuracy = accuracy_scores)

# En yüksek doğruluğa sahip modelin adını bulma
en_iyi_model <- accuracy_df[which.max(accuracy_df$Accuracy), "Model"]
en_iyi_dogruluk <- max(accuracy_df$Accuracy)

cat("En iyi model:", en_iyi_model, "\n")
cat("En iyi modelin doğruluğu:", en_iyi_dogruluk, "\n")


# Modeli kaydetme
if (en_iyi_model == "KNN") {
  saveRDS(tahmin, "en_iyi_model_knn.rds")
} else if (en_iyi_model == "NN") {
  saveRDS(nn, "en_iyi_model_nn.rds")
} else if (en_iyi_model == "RF") {
  saveRDS(kalp_rf, "en_iyi_model_rf.rds")
}else if (en_iyi_model == "SVM") {
  saveRDS(svm.linear, "en_iyi_model_svm.rds")
}else if (en_iyi_model == "GLM") {
  saveRDS(log.model, "en_iyi_model_glm.rds")
}

en_iyi_model_knn <- readRDS("en_iyi_model_knn.rds")
model <- class(en_iyi_model_knn)
# Yeni veri için tahmin yapma
yeni_veri <- data.frame(
  age = 63,
  sex = 1,
  cp = 3,
  trestbps = 145,
  chol = 233,
  fbs = 1,
  restecg = 0,
  thalach = 150,
  exang = 0,
  oldpeak = 2.3,
  slope = 0,
  ca = 0,
  thal = 1,
  target = 1
)
# Tahmin yapma
tahmin <- predict(model, yeni_veri, type = "class")

print("KNN ile tahmin:")
print(tahmin)
