import os
import sys


import time
import warnings
import pandas as pd
import streamlit as st
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import os
import sys
import time
import warnings
import pandas as pd
import streamlit as st

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel, GBTClassificationModel

warnings.filterwarnings('ignore')
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['HADOOP_HOME'] = r"C:\Users\Ahmed\Desktop\Project Big-Data\hadoop"
os.environ['hadoop.home.dir'] = r"C:\Users\Ahmed\Desktop\Project Big-Data\hadoop"

os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'

st.set_page_config(page_title="CTR Prediction App", layout="wide", initial_sidebar_state="expanded")

OUTPUT_PATH = r"C:\Users\Ahmed\Desktop\Project Big-Data\ProjectBigData"
MODEL_PATH  = OUTPUT_PATH + "\models/"
pipeline_path = OUTPUT_PATH + "\processed_datapipeline_model/"


models = ['Logistic Regression', 'Random Forest', 'GBT']

@st.cache_resource
def load_spark_and_models():
    spark = SparkSession.builder \
        .appName("Taobao_CTR_Modeling") \
        .master("local[1]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .getOrCreate()


    pipeline_model = PipelineModel.load(pipeline_path)
    lr_model = LogisticRegressionModel.load(MODEL_PATH + "logistic_regression")
    rf_model = RandomForestClassificationModel.load(MODEL_PATH + "random_forest")
    gbt_model = GBTClassificationModel.load(MODEL_PATH + "gbt")
    
    return spark, pipeline_model, lr_model, rf_model, gbt_model

spark, pipeline_model, lr_model, rf_model, gbt_model = load_spark_and_models()

df_result_models = pd.read_csv(r"C:\Users\Ahmed\Desktop\Project Big-Data\ProjectBigDatamodels_evaluation_results.csv")
best_model_name = df_result_models['Test F1-Score'].idxmax()
best_auc_score  = df_result_models['Test F1-Score'].max()

# Sidebar
with st.sidebar:
    st.markdown("# CTR Predict")
    st.markdown("Taobao Ad Click System")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Single Prediction", "Model Comparison"],
        label_visibility="collapsed"
    )

# PAGE 1 — Single Prediction
if page == "Single Prediction":
    st.markdown("# Single Prediction")
    st.markdown("Enter user & ad features to predict click probability")

    col1, col2, col3 = st.columns(3)

    with col1:
        user = st.text_input("User", value=12)
        adgroup_id = st.text_input("Adgroup", value=411858)
        time_stamp = st.text_input("Time Stamp", value=1494302581)
        cate_id = st.number_input("Category ID", value=4283)
        campaign_id = st.number_input("Campaign Id", value=49595)
        customer = st.number_input("Customer ID", value=75195)
        brand = st.text_input("Brand ID", value="311200")
        price = st.number_input("Price", value=198.0)

    with col2:
        cms_segid = st.number_input("CMS Seg ID", value=56)
        cms_group_id = st.number_input("CMS Group ID", value=8)
        shopping_level = st.selectbox("Shopping Level", [1, 2, 3])
        ad_historical_ctr = st.number_input("Ad Historical Ctr", value=0.0855)
        ad_impression_count = st.number_input("Ad Impression Count", value=152)
        user_total_clicks = st.number_input("User Total Clicks", value=0)
        user_total_impressions = st.number_input("User Total Impressions", value=3)
        user_ctr = st.number_input("User Ctr", value=3)

    with col3:
        pid = st.selectbox("PID (ex: 430548_1007)", ('430539_1007', '430548_1007'))
        occupation = st.selectbox("Occupation (0/1)", [0, 1])
        new_user_class_level = st.selectbox("New User Class Level", [0, 1, 2, 3, 4])
        hour = st.slider("Hour of Day", 0, 23, 12)
        day_of_week = st.slider("Day of Week", 1, 7, 7)
        final_gender_code = st.selectbox("Gender Code (1=Male, 2=Female)", [1, 2])
        age_level = st.selectbox("Age Level", [1, 2, 3, 4, 5, 6])
        pvalue_level = st.selectbox("P-value Level", [0, 1, 2, 3])
        is_weekend = st.selectbox("Is Weekend? (0=No, 1=Yes)", [0, 1])
        time_segment = st.selectbox("Time Segment", [0, 1, 2, 3])

    st.markdown("##### Select Model")
    selected_model = st.selectbox(
        "Model", ['Logistic Regression', 'Random Forest', 'GBT'],
        index=2,
        label_visibility="collapsed"
    )

    predict_btn = st.button(" Predict Click Probability", use_container_width=True, type="primary")

    if predict_btn:
        with st.spinner("Analyzing and extracting prediction..."):
            import time
            time.sleep(0.4)
            
            from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, LongType
            schema = StructType([
                StructField("user", IntegerType(), True),
                StructField("adgroup_id", IntegerType(), True),
                StructField("time_stamp", IntegerType(), True),
                StructField("pid", StringType(), True),
                StructField("cate_id", IntegerType(), True),
                StructField("campaign_id", IntegerType(), True),
                StructField("customer", IntegerType(), True),
                StructField("brand", StringType(), True),
                StructField("price", DoubleType(), True), # خلي بالك دي Double
                StructField("cms_segid", IntegerType(), True),
                StructField("cms_group_id", IntegerType(), True),
                StructField("final_gender_code", IntegerType(), True),
                StructField("age_level", IntegerType(), True),
                StructField("pvalue_level", IntegerType(), True),
                StructField("shopping_level", IntegerType(), True),
                StructField("occupation", IntegerType(), True),
                StructField("new_user_class_level", IntegerType(), True),
                StructField("hour", IntegerType(), True),
                StructField("day_of_week", IntegerType(), True),
                StructField("is_weekend", IntegerType(), True),
                StructField("time_segment", IntegerType(), True),
                StructField("ad_historical_ctr", DoubleType(), True),
                StructField("ad_impression_count", LongType(), True), # Long
                StructField("user_total_clicks", LongType(), True),   # Long
                StructField("user_total_impressions", LongType(), True),# Long
                StructField("user_ctr", DoubleType(), True)
            ])

            data = [(
                int(user),
                int(adgroup_id),
                int(time_stamp),
                str(pid), 
                int(cate_id),
                int(campaign_id),
                int(customer), 
                str(brand), 
                float(price),
                int(cms_segid), 
                int(cms_group_id), 
                int(final_gender_code), 
                int(age_level), 
                int(pvalue_level),
                int(shopping_level), 
                int(occupation), 
                int(new_user_class_level),
                int(hour), 
                int(day_of_week), 
                int(is_weekend),
                int(time_segment),
                float(ad_historical_ctr),
                int(ad_impression_count),
                int(user_total_clicks),
                int(user_total_impressions),
                float(user_ctr)
            )]

            input_df = spark.createDataFrame(data, schema=schema)

            transformed_df = pipeline_model.transform(input_df)

            if selected_model == 'Logistic Regression':
                model = lr_model
            elif selected_model == 'Random Forest':
                model = rf_model
            else:
                model = gbt_model

            prediction_df = model.transform(transformed_df)

            result = prediction_df.select("prediction", "probability").first()
            print(result)

            prob = result["probability"][1]

            click = prob >= 0.6
            color_of_prob = "green" if click else "red"

            predict = 'Click' if result["prediction"] else "Not Click"
            color_of_predict = "green" if result["prediction"] else "red"


            st.markdown("---")
            st.markdown(f"#### :{color_of_prob}[{'WILL CLICK'}]")
            st.markdown(f"## :{color_of_prob}[{prob*100:.1f}%]")
            st.markdown(f"###### But Prediction Is: :{color_of_predict}[{predict}]")
            st.markdown(f"**Click Probability** . {selected_model}")

elif page == "Model Comparison":
    st.markdown("## Model Comparison")
    st.markdown("Compare all 3 models · Select the best · Understand the results")
    st.markdown("Performance Metrics")

    st.table(df_result_models)

    col1, col2 = st.columns(2)

    with col1:
        df_plot = df_result_models[["Model", "Test Accuracy", "Test AUC-ROC", "Test F1-Score"]]

        df_melted = df_plot.melt(
            id_vars="Model",
            var_name="Metric",
            value_name="Score"
        )

        fig = px.bar(
            df_melted,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            text="Score",
            title="Model Comparison (Test Metrics)",
        )

        fig.update_traces(textposition='outside')

        st.plotly_chart(fig)


    with col2:
        st.markdown('Quick Model Selector')
        model_select = st.selectbox('Quick Model Selector', models)
    
        st.progress(df_result_models[df_result_models['Model'] == model_select]['Train AUC-ROC'].values[0], 
                    text=f"Train AUC-ROC: {df_result_models[df_result_models['Model'] == model_select]['Train AUC-ROC'].values[0]}")
        st.progress(df_result_models[df_result_models['Model'] == model_select]['Test AUC-ROC'].values[0], 
                    text=f"Test AUC-ROC: {df_result_models[df_result_models['Model'] == model_select]['Test AUC-ROC'].values[0]}")
        st.progress(df_result_models[df_result_models['Model'] == model_select]['Train Accuracy'].values[0], 
                    text=f"Train Accuracy{df_result_models[df_result_models['Model'] == model_select]['Train Accuracy'].values[0]}")
        st.progress(df_result_models[df_result_models['Model'] == model_select]['Test Accuracy'].values[0], 
                    text=f"Test Accuracy: {df_result_models[df_result_models['Model'] == model_select]['Test Accuracy'].values[0]}")
        st.progress(df_result_models[df_result_models['Model'] == model_select]['Test Precision'].values[0], 
                    text=f"Test Precision: {df_result_models[df_result_models['Model'] == model_select]['Test Precision'].values[0]}")
        st.progress(df_result_models[df_result_models['Model'] == model_select]['Test Recall'].values[0], 
                    text=f"Test Recall: {df_result_models[df_result_models['Model'] == model_select]['Test Recall'].values[0]}")
        st.progress(df_result_models[df_result_models['Model'] == model_select]['Test F1-Score'].values[0], 
                    text=f"Test F1-Score: {df_result_models[df_result_models['Model'] == model_select]['Test F1-Score'].values[0]}")

    st.markdown('FEATURE IMPORTANCE (RF vs GBT)')
    st.image(r"C:\Users\Ahmed\Desktop\Project Big-Data\ProjectBigData\charts\feature_importance.png", caption="FEATURE IMPORTANCE — RF vs GBT")

    st.markdown('CONFUSION MATRICES')
    st.image(r"C:\Users\Ahmed\Desktop\Project Big-Data\ProjectBigData\charts\confusion_matrices.png", caption="FEATURE IMPORTANCE — RF vs GBT")
