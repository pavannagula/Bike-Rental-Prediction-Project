rm(list=ls())

#Setting directory
setwd("C:/Users/npava/Desktop/Bike Rental Project/R")

#Loading Libraries
libraries = c("plyr","dplyr", "ggplot2","gplots","rpart","dplyr","DMwR","randomForest","usdm","corrgram","DataCombine","FNN",)
lapply(libraries, require, character.only = TRUE)
rm(libraries)

#Reading the csv file
bike_rental_data = read.csv("day.csv", header = T, sep = ',', na.strings = c(" ", "", "NA"))

############Data exploration##########

head(bike_rental_data)

names(bike_rental_data)

str(bike_rental_data)


######################################## MISSING VALUES ############################################
missing_values = sapply(bike_rental_data, function(x){sum(is.na(x))})

missing_values
#There are no missing values in the data set which can be confirmed after running the above line of codes.

####################################### Data Distribution ##########################################

#As the target variable is continuous variable we are going to use regression models for predictions
#So before we pass the data we need to check the Normal distribution of Data

#Distribution of Temp Variable
ggplot(bike_rental_data, aes_string(x = bike_rental_data$temp)) + 
   geom_histogram(fill="DarkSlateBlue", colour = "black", binwidth = 0.1) + geom_density() +
   theme_bw() + xlab("Temp") + ylab("Range") + ggtitle("Normal Distribution of Temperature Variable") +
   theme(text=element_text(size=10))

#Distribution of Atemp Variable
ggplot(bike_rental_data, aes_string(x = bike_rental_data$atemp)) + 
   geom_histogram(fill="DarkSlateBlue", colour = "black", binwidth = 0.1) + geom_density() +
   theme_bw() + xlab("Atemp") + ylab("Range") + ggtitle("Normal Distribution of Atemp Variable") +
   theme(text=element_text(size=10))

#Distribution of Humidity Variable
ggplot(bike_rental_data, aes_string(x = bike_rental_data$hum)) + 
   geom_histogram(fill="DarkSlateBlue", colour = "black", binwidth = 0.1) + geom_density() +
   theme_bw() + xlab("Humidity") + ylab("Range") + ggtitle("Normal Distribution of Humidity Variable") +
   theme(text=element_text(size=10))

#Distribution of Windspeed Variable
ggplot(bike_rental_data, aes_string(x = bike_rental_data$windspeed)) + 
   geom_histogram(fill="DarkSlateBlue", colour = "black", binwidth = 0.1) + geom_density() +
   theme_bw() + xlab("Windspeed") + ylab("Range") + ggtitle("Normal Distribution of Windspeed Variable") +
   theme(text=element_text(size=10))

####################################### Outlier Analysis ###########################################

#I have loaded the continuous or numeric varibales column names into numeric_var for detecting the outliers
numeric_var = colnames(bike_rental_data[,c("temp","atemp","hum","windspeed")])

for (i in 1:length(numeric_var))
{
  assign(paste0("gn",i), ggplot(aes_string(y = numeric_var[i]), data = bike_rental_data)+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=numeric_var[i])+
           ggtitle(paste("Box plot of",numeric_var[i])))
}

gridExtra::grid.arrange(gn1,gn3,gn2,gn4,ncol=2)


for (i in numeric_var) {
   print(i)
   val = bike_rental_data[,i][bike_rental_data[,i] %in% boxplot.stats(bike_rental_data[,i])$out]
   print(length(val))
   bike_rental_data = bike_rental_data[which(!bike_rental_data[,i] %in% val),]
}

########################################## feature selection ######################################################

#Check for multicollinearity using VIF
dataframe_numeric = bike_rental_data[,c("temp","atemp","hum","windspeed")]
vifcor(dataframe_numeric)
#The atemp variable is highly correlated with the temp which is also plotted using the correlation plot below.

#Check for collinearity using corelation graph
corrgram(bike_rental_data, order = F, upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

#Removing the correlated variables from the dataset

bike_rental_data <- subset(bike_rental_data, select = -c(instant,dteday,atemp,casual,registered))

rmExcept(keepers = "bike_rental_data")

head(bike_rental_data)

############################################ Model Deployment ######################################################

#Before passing the dataset into model we need to divide the data into train and test datasets
#Dividing the dataset into train and test datsets
set.seed(1234)
train_index = sample(1:nrow(bike_rental_data), 0.8 * nrow(bike_rental_data)) # dividing 80% of data into train dataset 
train = bike_rental_data[train_index,]
test = bike_rental_data[-train_index,]

# KNN IMPUTATION

#MAPE = 0.45%
#MAE = 9.20
#RMSE = 26.59
#ACCURACY = 99.55%

KNN_Predictions = knn.reg(train[, 1:11], test[, 1:11], train$cnt, k = 3)


#For calculating MAPE percentage
MAPE = function(Actual_val, Predicted_val){
   print(mean(abs((Actual_val - Predicted_val)/Actual_val)) * 100)
}

MAPE(test[,11], KNN_Predictions$pred)

regr.eval(trues = test[,11], preds = KNN_Predictions$pred, stats = c("mae","mse","rmse","mape"))

#Scatterplot between the Actual values and predicted values
ggplot(test, aes_string(x = test[,11], y = KNN_Predictions$pred)) + 
   geom_point() +
   theme_bw()+ ylab("Predicted Values") + xlab("Actual Values") + ggtitle("Scatter plot Analysis of KNN") + 
   theme(text=element_text(size=15)) + theme(plot.title = element_text(hjust = 0.5))


##Random Forest Model

#MAPE: 16.71%
#MAE: 502
#RMSE: 675
#Accuracy: 83.29%

#Train the data using random forest
rf_model = randomForest(cnt~., data = train, ntree = 500)

#Predict the test cases
rf_predictions = predict(rf_model, test[,-11])

#Calculate MAPE
regr.eval(trues = test[,11], preds = rf_predictions, stats = c("mae","mse","rmse","mape"))

MAPE(test[,11], rf_predictions)

#Scatterplot between the Actual values and predicted values from Random Forest Model
ggplot(test, aes_string(x = test[,11], y = rf_predictions)) + 
   geom_point() +
   theme_bw()+ ylab("Predicted Values") + xlab("Actual Values") + ggtitle("Scatter plot Analysis of Random Forest") + 
   theme(text=element_text(size=15)) + theme(plot.title = element_text(hjust = 0.5))


#LINEAR REGRESSION
#MAPE: 20.2%
#RMSE: 883
#MAE: 691
#Accuracy: 79.8%
#Adjusted R squared: 0.7887 
#F-statistic: 214.6

#Train the data using linear regression
lr_model = lm(formula = cnt~., data = train)

#Check the summary of the model
summary(lr_model)

#Predict the test cases
lr_predictions = predict(lr_model, test[,-11])

#Create dataframe for actual and predicted values
df = data.frame("actual"=test[,11], "pred"=lr_predictions)
df = cbind(df,lr_predictions)
head(df)

#Calculate MAPE
regr.eval(trues = test[,11], preds = lr_predictions, stats = c("mae","mse","rmse","mape"))
MAPE(test[,11], lr_predictions)

#Scatterplot between the Actual values and predicted values from Linear Regression Model
ggplot(test, aes_string(x = test[,11], y = lr_predictions)) + 
   geom_point() +
   theme_bw()+ ylab("Predicted Values") + xlab("Actual Values") + ggtitle("Scatter plot Analysis of Linear Regression") + 
   theme(text=element_text(size=15)) + theme(plot.title = element_text(hjust = 0.5))


#Predicting a sample data 
predict(rf_model,test[2,])
test[2,11]


########################################## Extra graphs  ###################

head(bike_rental_data)
ggplot(bike_rental_data, aes_string(x = bike_rental_data$weathersit)) +
   geom_bar(stat="count",fill =  "DarkSlateBlue") + theme_bw() +
   xlab("weather") + ylab('Count') +
   ggtitle("Bike Renatal Analysis") +  theme(text=element_text(size=15))


head(bike_rental_data)
ggplot(bike_rental_data, aes_string(x = bike_rental_data$workingday)) +
   geom_bar(stat="count",fill =  "DarkSlateBlue") + theme_bw() +
   xlab("workingday") + ylab('Count') +
   ggtitle("Bike Renatal Analysis") +  theme(text=element_text(size=15))


head(bike_rental_data)
ggplot(bike_rental_data, aes_string(x = bike_rental_data$holiday)) +
   geom_bar(stat="count",fill =  "DarkSlateBlue") + theme_bw() +
   xlab("holiday") + ylab('Count') +
   ggtitle("Bike Renatal Analysis") +  theme(text=element_text(size=15))


head(bike_rental_data)
ggplot(bike_rental_data, aes_string(x = bike_rental_data$season)) +
   geom_bar(stat="count",fill =  "DarkSlateBlue") + theme_bw() +
   xlab("season") + ylab('Count') +
   ggtitle("Bike Renatal Analysis") +  theme(text=element_text(size=15))


head(bike_rental_data)
ggplot(bike_rental_data, aes_string(x = bike_rental_data$month)) +
   geom_bar(stat="count",fill =  "DarkSlateBlue") + theme_bw() +
   xlab("month") + ylab('Count') +
   ggtitle("Bike Renatal Analysis") +  theme(text=element_text(size=15))


head(bike_rental_data)
ggplot(bike_rental_data, aes_string(x = bike_rental_data$yr)) +
   geom_bar(stat="count",fill =  "DarkSlateBlue") + theme_bw() +
   xlab("year") + ylab('Count') +
   ggtitle("Bike Renatal Analysis") +  theme(text=element_text(size=15))



