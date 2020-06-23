from pyspark import SparkContext
logFile = "file:////Users/vantinhchu/Documents/01_htw/semester_4/Pycharm_project/ICW/README.md"
sc = SparkContext("local", "first app")
logData = sc.textFile(logFile).cache()
numAs = logData.filter(lambda s: 'a' in s).count()
numBs = logData.filter(lambda s: 'b' in s).count()
print("Lines with a: %i, lines with b: %i" % (numAs, numBs))