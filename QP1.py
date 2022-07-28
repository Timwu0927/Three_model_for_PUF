from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier, MultilayerPerceptronClassifier
import json
import time

spark = SparkSession.builder \
    .appName("Ting Spark Intro") \
    .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

df_train = spark.read.load("/home/acr20tw/com6012/ScalableML/assign1/XOR_Arbiter_PUFs/5xor_128bit/train_5xor_128dim.csv",format = 'csv', header = "false",inferSchema = "true").cache()
df_test = spark.read.load("/home/acr20tw/com6012/ScalableML/assign1/XOR_Arbiter_PUFs/5xor_128bit/test_5xor_128dim.csv",format = 'csv', header = "false",inferSchema = "true").cache()

df_train_sample = df_train.sample(0.01)
df_test_sample = df_test.sample(0.01)
df_train_sample = df_train_sample.cache()
df_test_sample = df_test_sample.cache()

def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])

transformed_train = transData(df_train)
transformed_test = transData(df_test)
transformed_train = transformed_train.cache()
transformed_test = transformed_test.cache()

transformed_train_sample = transData(df_train_sample)
transformed_test_sample = transData(df_test_sample)
transformed_train_sample = transformed_train_sample.cache()
transformed_test_sample = transformed_test_sample.cache()


update_func = (F.when(F.col("label") == -1,0).otherwise(F.col("label")))
df_train  = transformed_train.withColumn("label",update_func).cache()
df_test = transformed_test.withColumn("label",update_func).cache()
df_train_sample  = transformed_train_sample.withColumn("label",update_func).cache()
df_test_sample = transformed_test_sample.withColumn("label",update_func).cache()


evaluator = BinaryClassificationEvaluator\
      (labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC") 
      
evaluator2 = MulticlassClassificationEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="accuracy")




print('----------------1% of Data Random Forest Model------------------------')

rf = RandomForestClassifier(labelCol="label", featuresCol="features")
pipeline_rf = Pipeline(stages = [rf])

paramGrid_rf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [1,5,10]) \
    .addGrid(rf.maxDepth, [1,5,10]) \
    .addGrid(rf.maxBins, [2,10,20]) \
    .build()

cv_rf = CrossValidator(estimator=pipeline_rf,
                          estimatorParamMaps=paramGrid_rf,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=5)

cvModel_rf = cv_rf.fit(df_train_sample)
predictions = cvModel_rf.transform(df_test_sample)

accuracy = evaluator2.evaluate(predictions)
auc = evaluator.evaluate(predictions)

print("Random Forest 1% Data Accuracy",accuracy)
print("Random Forest 1% Data AUC",auc)


paramDict_rf = {param[0].name: param[1] for param in cvModel_rf.bestModel.stages[-1].extractParamMap().items()}
print(json.dumps(paramDict_rf, indent = 4))
print('---------------------------------------------------------------------')


print('----------------1% of Data Logistic Regression------------------------')

lr = LogisticRegression(labelCol="label", featuresCol="features")
pipeline_lr = Pipeline(stages = [lr])

param_grid_lr = (ParamGridBuilder().addGrid(lr.elasticNetParam, [0,0.5,1.0]).addGrid(lr.regParam, [0.5,0.1, 0.01]).addGrid(lr.maxIter,[5,10,20]).build())

cv_lr = CrossValidator(estimator=pipeline_lr, estimatorParamMaps=param_grid_lr, evaluator=BinaryClassificationEvaluator(), numFolds=5)

cvModel_lr = cv_lr.fit(df_train_sample)
predictions = cvModel_lr.transform(df_test_sample)

accuracy = evaluator2.evaluate(predictions)
auc = evaluator.evaluate(predictions)

print("Logistic Regression 1% Data Accuracy",accuracy)
print("Logistic Regression 1% Data AUC",auc)
predictions.show(10)

paramDict_lr = {param[0].name: param[1] for param in cvModel_lr.bestModel.stages[-1].extractParamMap().items()}
print(json.dumps(paramDict_lr, indent = 4))
print('---------------------------------------------------------------------')

print('------------------------------1% of Data NN---------------------------')

nn = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features", seed=16269)
pipeline_nn = Pipeline(stages = [nn])

paramGrid_nn = ParamGridBuilder() \
            .addGrid(nn.layers, [[128,20,5,2], 
                                  [128,40,10,2],
                                  [128,40,20,2]])\
            .addGrid(nn.maxIter,[20,50,100])\
            .addGrid(nn.stepSize,[0.1,0.2,0.3])\
            .build()
            
cv_nn = CrossValidator(estimator=pipeline_nn,
                          estimatorParamMaps=paramGrid_nn,
                          evaluator=evaluator2,
                          numFolds=5)
                          
cvModel_nn = cv_nn.fit(df_train_sample)
predictions = cvModel_nn.transform(df_test_sample)

accuracy = evaluator2.evaluate(predictions)
auc = evaluator.evaluate(predictions)

print("NN 1% Data Accuracy",accuracy)
print("NN 1% Data AUC",auc)

paramDict_nn = {param[0].name: param[1] for param in cvModel_nn.bestModel.stages[-1].extractParamMap().items()}
print(json.dumps(paramDict_nn, indent = 4))
print('---------------------------------------------------------------------')
print('---------------------------------------------------------------------')
print('---------------------------------------------------------------------')
print('---------------------------------------------------------------------')



print('-------------------100% Data Random Forest Model----------------------')
rf = RandomForestClassifier(labelCol="label", featuresCol="features", maxDepth=paramDict_rf['maxDepth'],numTrees=paramDict_rf['numTrees'],maxBins=paramDict_rf['maxBins'],seed=16269)
pipeline_rf = Pipeline(stages = [rf])


start_train_time = time.time()
pipeline_rf_model = pipeline_rf.fit(df_train)
end_train_time = time.time()
print("Total Training Time of Random Forest Model with " + str(sc.master)+ " cores",(end_train_time - start_train_time))

start_test_time = time.time()
predictions = pipeline_rf_model.transform(df_test)
end_test_time = time.time()
print("Total Testing Time of Random Forest Model with "+ str(sc.master)+ "cores",(end_test_time - start_test_time))

accuracy = evaluator2.evaluate(predictions)
auc = evaluator.evaluate(predictions)

print("Random Forest 100% Data Accuracy with "+ str(sc.master)+ "cores",accuracy)
print("Random Forest 100% Data AUC with "+ str(sc.master)+ "cores",auc)
print('---------------------------------------------------------------------')


print('-------------------100% Data Logistic Regression Model----------------------')

lr = LogisticRegression(labelCol="label", featuresCol="features", elasticNetParam=paramDict_lr['elasticNetParam'],regParam=paramDict_lr['regParam'],maxIter=paramDict_lr['maxIter'])
pipeline_lr = Pipeline(stages = [lr])

start_train_time = time.time()
pipeline_lr_model = pipeline_lr.fit(df_train)
end_train_time = time.time()
print("Total Training Time of Logistic Regression Model with "+ str(sc.master)+ "cores", (end_train_time - start_train_time))

start_test_time = time.time()
predictions = pipeline_lr_model.transform(df_test)
end_test_time = time.time()
print("Total Testing Time of Logistic Regression Model with "+ str(sc.master)+ "cores", (end_test_time - start_test_time))

accuracy = evaluator2.evaluate(predictions)
auc = evaluator.evaluate(predictions)
print("Logistic Regression 100% Data Accuracy with "+ str(sc.master)+ "cores",accuracy)
print("Logistic Regression 100% Data AUC with "+ str(sc.master)+ "cores",auc)
print('--------------------------------------------------------------------')


print('--------------------------100% Data NN Model-------------------------')

nn = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features", layers=paramDict_nn['layers'],stepSize=paramDict_nn['stepSize'],maxIter=paramDict_nn['maxIter'],seed=16269)
pipeline_nn = Pipeline(stages = [nn])

start_train_time = time.time()
pipeline_nn_model = pipeline_nn.fit(df_train)
end_train_time = time.time()
print("Total Training Time of NN Model with "+ str(sc.master)+ "cores", (end_train_time - start_train_time))

start_test_time = time.time()
predictions = pipeline_nn_model.transform(df_test)
end_test_time = time.time()
print("Total Testing Time of NN Model with "+ str(sc.master)+ "cores",(end_test_time - start_test_time))

accuracy = evaluator2.evaluate(predictions)
auc = evaluator.evaluate(predictions)
print("NN 100% Data Accuracy with "+ str(sc.master)+ "cores",accuracy)
print("NN 100% Data AUC with "+ str(sc.master)+ "cores",auc)
print('---------------------------------------------------------------------')


