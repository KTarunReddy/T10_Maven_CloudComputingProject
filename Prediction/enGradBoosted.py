from dateutil.parser import parse
from dateutil.parser import parse
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from time import time

#loading trainHistory table with generated Features
from pyspark import SparkContext
sc = SparkContext(appName = "test")
trainHistory = sc.textFile("D:/CloudComputing/CourseProject/Data/Merge/Reduction/FeatureGeneration/Prediction/train_history_features.csv")
header = trainHistory.filter(lambda line: "CUSTOMER_ID" in line)
trainHistory= trainHistory.subtract(header)
schemaString = "customer_id offer_id chain_id market_id repeat_trips offer_date category_id min_qty company_id offer_value brand_id purchase_times_company purchase_value_company purchase_quantity_company repeat_customer times_company_180 times_company_60 times_company_30 purchase_times_ccb purchase_times_category amount_spent_category bought_category_30 chain_visit_freq ratio_returned_bought_cc"
fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
fields[4].dataType = FloatType()
fields[5].dataType = TimestampType()
fields[7].dataType = FloatType()
fields[9].dataType = FloatType()
fields[11].dataType = FloatType()
fields[12].dataType = FloatType()
fields[13].dataType = FloatType()
fields[14].dataType = IntegerType()
fields[15].dataType = FloatType()
fields[16].dataType = FloatType()
fields[17].dataType = FloatType()
fields[18].dataType = FloatType()
fields[19].dataType = FloatType()
fields[20].dataType = FloatType()
fields[21].dataType = FloatType()
fields[22].dataType = FloatType()
fields[23].dataType = FloatType()

schema = StructType(fields)
trainHistory = trainHistory.map(lambda l: l.split(",")).map(lambda p: (p[0],p[1],p[2],p[3],float(p[4]),parse(p[5]),p[6],float(p[7]),p[8],float(p[9]),p[10],float(p[11]),float(p[12]),float(p[13]),int(p[14]),float(p[15]),float(p[16]),float(p[17]),float(p[18]),float(p[19]),float(p[20]),float(p[21]),float(p[22]),float(p[23])))
sqlContext = SQLContext(sc)
trainHistoryDF = sqlContext.createDataFrame(trainHistory, schema) 

#loading testHistory table 

testHistory = sc.textFile("D:/CloudComputing/CourseProject/Data/Merge/Reduction/FeatureGeneration/Prediction/test_history_features.csv")
header = testHistory.filter(lambda line: "CUSTOMER_ID" in line)
testHistory= testHistory.subtract(header)
schemaString = "customer_id offer_id chain_id market_id offer_date category_id min_qty company_id offer_value brand_id purchase_times_company purchase_value_company purchase_quantity_company times_company_180 times_company_60 times_company_30 purchase_times_ccb purchase_times_category amount_spent_category bought_category_30 chain_visit_freq ratio_returned_bought_cc"
fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
fields[4].dataType = TimestampType()
fields[6].dataType = FloatType()
fields[8].dataType = FloatType()
fields[10].dataType = FloatType()
fields[11].dataType = FloatType()
fields[12].dataType = FloatType()
fields[13].dataType = FloatType()
fields[14].dataType = FloatType()
fields[15].dataType = FloatType()
fields[16].dataType = FloatType()
fields[17].dataType = FloatType()
fields[18].dataType = FloatType()
fields[19].dataType = FloatType()
fields[20].dataType = FloatType()
fields[21].dataType = FloatType()

schema = StructType(fields)
testHistory = testHistory.map(lambda l: l.split(",")).map(lambda p: (p[0],p[1],p[2],p[3],parse(p[4]),p[5],float(p[6]),p[7],float(p[8]),p[9],float(p[10]),float(p[11]),float(p[12]),float(p[13]),float(p[14]),float(p[15]),float(p[16]),float(p[17]),float(p[18]),float(p[19]),float(p[20]),float(p[21])))
testHistoryDF = sqlContext.createDataFrame(testHistory, schema) 


#parsing data into LabeledPoint object

parsedTrainData= trainHistoryDF.rdd.map(lambda line:LabeledPoint(line[14],[line[7],line[9],line[11],line[12],line[13],line[15],line[16],line[17],line[18],line[19],line[20],line[21],line[22],line[23]]))

#partitioning the training data into training and testing sets

#parsedTrainData, parsedTestData = parsedTrainData.randomSplit([0.7, 0.3], seed = 1245)

parsedTestData= testHistoryDF.rdd.map(lambda line:LabeledPoint(line[0],[line[6],line[8],line[10],line[11],line[12],line[13],line[14],line[15],line[16],line[17],line[18],line[19],line[20],line[21]]))

# Building the model
t0 = time()
tree_model = GradientBoostedTrees.trainClassifier(parsedTrainData,
    categoricalFeaturesInfo={}, numIterations=4)
tt = time() - t0
print "Classifier trained in {} seconds".format(round(tt,3))

# evaluating the model on testing data
predictions = tree_model.predict(parsedTestData.map(lambda p: p.features))
labels_and_preds = parsedTestData.map(lambda p: p.label).zip(predictions)

t0 = time()
test_accuracy = labels_and_preds.filter(lambda (v, p): v == p).count() / float(parsedTestData.count())
tt = time() - t0

print "Prediction made in {} seconds. Test accuracy is {}".format(round(tt,3),test_accuracy)

# Instantiate metrics object
predictions.saveAsTextFile("D:/CloudComputing/CourseProject/Data/Merge/Reduction/FeatureGeneration/Prediction/predictionOutput/predictionFile")
#metrics = BinaryClassificationMetrics(labels_and_preds)

# Area under precision-recall curve
#print("Area under PR = %s" % metrics.areaUnderPR)

# Area under ROC curve
#print("Area under ROC = %s" % metrics.areaUnderROC)

print "Learned classification tree model:"
print tree_model.toDebugString()
