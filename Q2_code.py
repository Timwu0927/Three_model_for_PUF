
from functools import reduce  # For Python 3.x
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import split, regexp_extract 
from pyspark.sql.functions import date_format
from pyspark.sql.functions import dayofweek
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder \
        .master("local[4]") \
        .appName("assign2_Q2") \
        .config("spark.local.dir","/fastdata/acr20tw") \
        .getOrCreate()
        
sc = spark.sparkContext
sc.setLogLevel("ERROR") 

#the function to union whole dataframe
def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)
    
# load in ratings data and tags data 
ratings = spark.read.load('../Data/ml-25m/ratings.csv', format = 'csv', inferSchema = "true", header = "true").cache()
tags = spark.read.load('../Data/ml-25m/tags.csv', format = 'csv', inferSchema = "true", header = "true").cache()

#split the ratings data randomly into 5 folders (seed is the last 5 nums on student card) 
(fold1,fold2,fold3,fold4,fold5) = ratings.randomSplit([0.2,0.2,0.2,0.2, 0.2],seed = 16269)
fold1 = fold1.cache()
fold2 = fold2.cache()
fold3 = fold3.cache()
fold4 = fold4.cache()
fold5 = fold5.cache()
fold_list = [fold1,fold2,fold3,fold4,fold5] #fold_list if for the cross validation later

#assign lists for storing the final result
als_hot_result = []
als_cool_result = []
als2_hot_result = []
als2_cool_result = []
top_tags_result = []
least_tags_result = []


for i , testi in enumerate(fold_list):
  train = []
  test = []
  train.append(fold_list[:i] + fold_list[i+1:])
  test.append(testi)
  fold1 = train[0][0]
  fold2 = train[0][1]
  fold3 = train[0][2]
  fold4 = train[0][3]
  test_fold = test[0] 
  
  #test_fold = fold5
  test_fold.cache()
  test_fold_userid = test_fold.select("userId").rdd.flatMap(lambda x: x).collect()
  
  train_df = unionAll(fold1,fold2,fold3,fold4)
  train_df = train_df.cache()
  
  #use count to find out the top 10% and least 10% as Hot and Cool users
  train_hot_temp = train_df.groupBy('userId').count().sort('count',ascending = False).cache()
  train_hot_temp = train_hot_temp.limit(int(0.1 * train_hot_temp.count())).cache()
  train_hot_userid = train_hot_temp.select("userId").rdd.flatMap(lambda x: x).collect()
  
  train_cool_temp = train_df.groupBy('userId').count().sort('count',ascending = True).cache()
  train_cool_temp = train_cool_temp.limit(int(0.1 * train_cool_temp.count())).cache()
  train_cool_userid = train_cool_temp.select("userId").rdd.flatMap(lambda x: x).collect()
  
  #find the Hot and Cool users in test data by using filter 
  test_hot_fold = test_fold.filter(test_fold.userId.isin(train_hot_userid))
  test_cool_fold = test_fold.filter(test_fold.userId.isin(train_cool_userid))
  
  #train Model and get the itemFactor to use later in the K means model
  als = ALS(userCol="userId", itemCol="movieId", seed=16269, coldStartStrategy="drop")
  model = als.fit(train_df)
  dfItemFactors = model.itemFactors
  #dfItemFactors.show()
  #print(als.getAlpha(),als.getMaxIter(),als.getRank(),als.getRegParam())
  
  #calculate the RMSE of hot users and cool users
  predictions_hot = model.transform(test_hot_fold)
  evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
  rmse = evaluator.evaluate(predictions_hot)
  als_hot_result.append(rmse)
  print("ALS Model Test Fold" + str(i+1) +" Hot_Root-mean-square error = " + str(rmse))
  
  predictions_cool = model.transform(test_cool_fold)
  evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
  rmse = evaluator.evaluate(predictions_cool)
  als_cool_result.append(rmse)
  print("ALS Model Test Fold" + str(i+1) +" Cool_Root-mean-square error = " + str(rmse))
  
  #train 2nd Als model (changge some parameters)
  als2 = ALS(userCol="userId", itemCol="movieId", seed=16269, regParam = 0.1,rank =20,coldStartStrategy="drop")
  model2 = als2.fit(train_df)
  #print(als2.getAlpha(),als2.getMaxIter(),als2.getRank(),als2.getRegParam())
  
  #calculate the RMSE of hot users and cool users
  predictions_hot = model2.transform(test_hot_fold)
  evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
  rmse = evaluator.evaluate(predictions_hot)
  als2_hot_result.append(rmse)
  print("ALS2 Model Test Fold" + str(i+1) +" Hot_Root-mean-square error = " + str(rmse))
  
  predictions_cool = model2.transform(test_cool_fold)
  evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
  rmse = evaluator.evaluate(predictions_cool)
  als2_cool_result.append(rmse)
  print("ALS2 Model Test Fold" + str(i+1) +" Cool_Root-mean-square error = " + str(rmse))
  
  #doing K means and the k is set as 10
  numK = 10 
  kmeans = KMeans().setK(numK).setSeed(16269)
  model = kmeans.fit(dfItemFactors)   
  predictions = model.transform(dfItemFactors)
  #count the prediction to get the top 2 largest clusters according to their size
  cluster_count = predictions.groupBy('prediction').count().sort('count',ascending = False).limit(2).cache()
  cluster_top = cluster_count.select("prediction").rdd.flatMap(lambda x: x).collect()
  
  top2_df1 = predictions.filter(predictions.prediction.isin(cluster_top[0]))
  top2_df2 = predictions.filter(predictions.prediction.isin(cluster_top[1]))
  movieid1 = top2_df1.select("id").rdd.flatMap(lambda x: x).collect()
  movieid2 = top2_df2.select("id").rdd.flatMap(lambda x: x).collect()
  
  #find the top and least tags
  tags_df1 = tags.filter(tags.movieId.isin(movieid1)).cache()
  tags_df2 = tags.filter(tags.movieId.isin(movieid2)).cache()
  top_tags1 = tags_df1.groupBy('tag').count().sort('count',ascending = False).limit(1).cache()
  top_tags2 = tags_df2.groupBy('tag').count().sort('count',ascending = False).limit(1).cache()
  tags_name1 = top_tags1.select("tag").rdd.flatMap(lambda x: x).collect()
  tags_name2 = top_tags2.select("tag").rdd.flatMap(lambda x: x).collect()
  #print(tags_name1,tags_name2)
  top_tags_result.append(tags_name1)
  top_tags_result.append(tags_name2)
  
  least_tags1 = tags_df1.groupBy('tag').count().sort('count',ascending = True).limit(1).cache()
  least_tags2 = tags_df2.groupBy('tag').count().sort('count',ascending = True).limit(1).cache()
  #least_tags.show(10,False)
  least_tags_name1 = least_tags1.select("tag").rdd.flatMap(lambda x: x).collect()
  least_tags_name2 = least_tags2.select("tag").rdd.flatMap(lambda x: x).collect()
  #print(least_tags_name1,least_tags_name2)
  least_tags_result.append(least_tags_name1)
  least_tags_result.append(least_tags_name2)

print("----------------------Q1-------------------------")
all_rmse = []
all_rmse.append(als_hot_result)
all_rmse.append(als_cool_result)
all_rmse.append(als2_hot_result)
all_rmse.append(als2_cool_result)
all_rmse = np.array(all_rmse)
all_rmse.reshape(4,5)
print(als_hot_result)
print(als_cool_result)
print(als2_hot_result)
print(als2_cool_result)
df_rmse = pd.DataFrame(all_rmse,index = ['Standard_ALS_Hot', 'Standard_ALS_Cool', 'Setting2_ALS_Hot','Setting2_ALS_Cool'], columns=['split1', 'split2', 'split3','split4','split5'])
print(df_rmse)

width = 0.2
x_axis = ['split1','split2','split3','split4','split5']
x_len = np.arange(5)
plt.figure(figsize=(10,8))
plt.bar(x_len,als_hot_result,width,color = 'green',label = 'ALS_Hot')
plt.bar(x_len+width,als_cool_result,width,color = 'blue',label='ALS_Cool')
plt.bar(x_len+2*width,als2_hot_result,width,color = 'gray',label='ALS2_Hot')
plt.bar(x_len+3*width,als2_cool_result,width,color = 'yellow',label='ALS2_Cool')
plt.xticks( x_len+ width, x_axis)
plt.title("ALS RMSE")
plt.xlabel("Each Split")
plt.ylabel("RMSE Num")
plt.savefig("../Output/Q2p2_figA.png")

print("--------------------------------------------------")
print("----------------------Q2-------------------------")
print(top_tags_result)
print(least_tags_result)
temp_top1=[]
temp_top2=[]
temp_least1=[]
temp_least2=[]
tags_final =[]
for i in range(len(top_tags_result)):
  if i%2 ==0:
    temp_top1.append(top_tags_result[i])
  else:
    temp_top2.append(top_tags_result[i])
    
for j in range(len(least_tags_result)):
  if j%2 ==0:
    temp_least1.append(least_tags_result[j])
  else:
    temp_least2.append(least_tags_result[j])    




tags_final.append(temp_top1)
tags_final.append(temp_least1)
tags_final.append(temp_top2)
tags_final.append(temp_least2)
#print(tags_final)
df_tags = pd.DataFrame(tags_final,index= ['1st Cluster Top Tag', '1st Cluster Least Tag', '2nd Cluster Top Tag','2nd Cluster Least Tag'], columns=['split1', 'split2', 'split3','split4','split5'])
print(df_tags)
print("--------------------------------------------------")




