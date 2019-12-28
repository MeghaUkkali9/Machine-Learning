#Name:                    Megha Ukkali
#Assignment number:       Project 3
#Date:                    11/25/2019

install.packages("RWeka")
install.packages("sparklyr")
install.packages("C50")


library(RWeka)
library(sparklyr)
library(C50)


spark_install()
sc <- spark_connect(master = "local")

PhishingData <- read.arff("C:/Users/Sony/Downloads/PhishingData.arff")

View(PhishingData)
summary(PhishingData)
fix(PhishingData)
str(PhishingData)

PhishingData_arff_tbl <- copy_to(sc, PhishingData, name = "PhishingData_arff_tbl", overwrite = TRUE)

PhishingData_arff_tbl


partitions <- PhishingData_arff_tbl %>%
  sdf_random_split(training = 0.7, test = 0.3, seed = 5111)


PhishingData_training <- partitions$training
PhishingData_test <- partitions$test

decision_tree_model <- PhishingData_training %>%
  ml_decision_tree(Result ~ ., type = "classification")


prediction = ml_predict(decision_tree_model, PhishingData_test)
ml_multiclass_classification_evaluator(prediction)

set.seed(1111)
Phishing <- PhishingData[order(runif(1353)), ]

Phishing_training <- Phishing[1:947,]
Phishing_test <- Phishing[948:1353,]


input <- Phishing_training[,1:9]
output <- Phishing_training[,10]

summary(input)
summary(output)


model1 <- C5.0(input, output, control = C5.0Control(noGlobalPruning = TRUE,minCases=1)) 
plot(model1, main="C5.0 Decision Tree - Unpruned, min=1")
model1
summary(model1)


model2 <- C5.0(input, output, control = C5.0Control(noGlobalPruning = FALSE))
plot(model2, main="C5.0 Decision Tree -Pruned")
model2
summary(model2)

rules_model <- C5.0(input, output, rules=TRUE)
summary(rules_model)
