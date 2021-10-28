import argparse
import pyspark.sql.functions as f
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import IntegerType

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--crime_df_path', dest='crime_df_path', required=True)
    args_parser.add_argument('--codes_df_path', dest='codes_df_path', required=True)
    args_parser.add_argument('--output_path', dest='output_path', required=True)
    args = args_parser.parse_args()

    spark = SparkSession.builder.appName('hw').getOrCreate()

    df = spark.read.format('csv').option('header', 'true').load(args.crime_df_path)
    df = df.na.drop(subset=['DISTRICT'])
    df = df.withColumn('OFFENSE_CODE', df['OFFENSE_CODE'].cast(IntegerType()))
    df = df.drop_duplicates()
    # print(df.count())

    df_types = spark.read.format('csv').option('header', 'true').load(args.codes_df_path)
    df_types = df_types.orderBy('CODE').drop_duplicates(subset=['CODE'])
    df_types = df_types.withColumn('crime_type', f.split(df_types['NAME'], '-').getItem(0))
    # print(df_types.count())

    df = df.join(df_types, df.OFFENSE_CODE == df_types.CODE, how='left')

    cm = df.groupBy(['DISTRICT', 'YEAR', 'MONTH']).agg(f.count('INCIDENT_NUMBER').alias('offence_cnt')).groupBy(
        'DISTRICT').agg(f.expr('percentile_approx(offence_cnt, 0.5)').alias('crimes_monthly'))

    fct = df.groupBy('DISTRICT', 'crime_type').count()
    window = Window.partitionBy('DISTRICT').orderBy(f.desc('count'))
    fct = fct.withColumn('order', f.row_number().over(window)).where(
        (f.col('order') == 1) | (f.col('order') == 2) | (f.col('order') == 3))
    fct = fct.groupby('DISTRICT').agg(f.concat_ws(', ', f.collect_list(fct.crime_type)).alias('frequent_crime_types'))

    cll = df.groupBy('DISTRICT').agg(f.count('INCIDENT_NUMBER').alias('crimes_total'), f.mean('Lat').alias('lat'),
                                     f.mean('Long').alias('lng'))

    result = cll.join(cm, 'DISTRICT')
    result = result.join(fct, 'DISTRICT').orderBy('crimes_total', ascending=False)
    result = result.withColumnRenamed('DISTRICT', 'district')
    result = result.select(['district', 'crimes_total', 'crimes_monthly', 'frequent_crime_types', 'lat', 'lng'])
    result.write.parquet(args.output_path, mode='overwrite')
