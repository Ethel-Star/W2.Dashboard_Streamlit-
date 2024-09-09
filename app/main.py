import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils import DataUtils
from io import BytesIO

# Define the file path
data_path = r'E:\2017.Study\Tenx\Week-2\Data\data\Week2_challenge_data_source(CSV).csv'

# Load the data
df = pd.read_csv(data_path, encoding='ISO-8859-1')

# Initialize DataUtils
data_utils = DataUtils(df)

# Set Streamlit page configuration
st.set_page_config(page_title="TellCo Telecom Analysis", layout="wide")

# Check and handle missing values, outliers
missing_summary = data_utils.check_missing_values()
df = data_utils.handle_missing_values()
outliers = data_utils.detect_outliers()

# Displaying outliers information
for column, outlier_indices in outliers.items():
    print(f"Outliers in column '{column}': {outlier_indices[:5]}")  # Show first 5 outlier indices

# Handle and remove outliers
df = data_utils.fix_outliers()
df = data_utils.remove_outliers()
df = data_utils.convert_bytes_to_megabytes()

# Dashboard title and introduction
st.title("TellCo Telecom Analysis: Uncovering Growth Opportunities")
st.markdown("""
## Introduction
This dashboard provides a comprehensive analysis of TellCo Telecom's data to uncover potential growth opportunities.
By analyzing customer behavior, user engagement, and network performance, we provide insights into future business directions.

## Business Objective
Identify growth opportunities by analyzing TellCo's customer data, focusing on user behavior, engagement, network performance, and satisfaction.
""")

# Sidebar for task selection
st.sidebar.title("Select Analysis Task")
task = st.sidebar.selectbox(
    "Choose a task:",
    ["Project Summary", "User Overview Analysis", "User Engagement Analysis", "Experience Analytics", "Satisfaction Analysis"]
)

# Define the logic for the task selected
if task == "Project Summary":
    st.markdown("""
    ## Project Summary
    ### Overview of Analytical Tasks:
    - **User Overview Analysis:** Identifying top handsets and manufacturers.
    - **User Behavior Analysis:** Analyzing xDR sessions, session duration, and data usage.
    - **User Engagement Analysis:** Clustering and segmentation based on session metrics.
    - **Experience Analytics:** Evaluating device and network performance impact on user experience.
    - **Satisfaction Analysis:** Predicting satisfaction scores based on user engagement and experience metrics.
    """)

elif task == "User Overview Analysis":
    st.subheader("User Overview Analysis")
    selection = st.sidebar.radio("Select Section", ["User Overview", "User Behavior", "EDA"])

    if selection == "User Overview":
        # Top 10 handsets used by customers
        st.subheader("Top 10 Handsets Used by Customers")
        top_handsets = df['Handset Type'].value_counts().head(10)
        st.bar_chart(top_handsets)

        # Plot a pie chart for the top 10 handsets
        st.subheader("Pie Chart of Top 10 Handsets")
        top_handsets_pie = top_handsets
        fig, ax = plt.subplots(figsize=(10, 7))  # Adjust size of the figure

        wedges, texts, autotexts = ax.pie(
            top_handsets_pie,
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.get_cmap('tab20').colors
        )

        # Customize pie chart
        ax.set_aspect('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

        # Adjust pie chart and legend to avoid overlap
        plt.tight_layout()
        
        # Add legend outside the pie chart
        plt.legend(
            wedges,
            top_handsets_pie.index,
            loc='upper left',
            bbox_to_anchor=(1, 1),  # Place legend outside the plot
            fontsize='small',  # Smaller font size
            title="Handset Types"  # Optional: Add a title to the legend
        )

        st.pyplot(fig)

        # Top 5 handsets per top 3 handset manufacturers
        st.subheader("Top 5 Handsets Per Top 3 Handset Manufacturers")
        top_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
        
        # Create subplots in a single row
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns
        axes = axes.flatten()  # Flatten array for easy iteration
        
        for i, manufacturer in enumerate(top_manufacturers.index):
            subset = df[df['Handset Manufacturer'] == manufacturer]
            top_handsets_per_manufacturer = subset['Handset Type'].value_counts().head(5)

            wedges, texts, autotexts = axes[i].pie(
                top_handsets_per_manufacturer,
                labels=top_handsets_per_manufacturer.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=plt.get_cmap('tab20').colors
            )
            
            axes[i].set_title(f"Top 5 Handsets for {manufacturer}")
            axes[i].set_aspect('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

        plt.tight_layout()
        st.pyplot(fig)

        # Marketing recommendations
        st.subheader("Marketing Recommendations")
        st.markdown("""
        - Prioritize iOS-exclusive features for Apple consumers, who dominate device usage.
        - Launch campaigns focusing on popular Samsung models like Galaxy S8 and Galaxy J5.
        - Address the "undefined" handset category for better targeting and data accuracy.
        - Focus on mid-range smartphones for Huawei customers, with products like P20 Lite and Y6 2018.
        """)
    elif selection == "User Behavior":
        st.subheader("User Behavior Analysis")

        # Convert bytes to megabytes
        df = data_utils.convert_bytes_to_megabytes()

        # Aggregate data per application
        applications = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']
        aggregated_data = pd.DataFrame()

        for app in applications:
            app_dl_col = f'{app} DL (MB)'
            app_ul_col = f'{app} UL (MB)'

            aggregated_data[f'{app} xDR Sessions'] = df.groupby('IMSI').size()
            aggregated_data[f'{app} Session Duration (ms)'] = df.groupby('IMSI')['Dur. (ms)'].sum()
            aggregated_data[f'{app} Total DL (MB)'] = df.groupby('IMSI')[app_dl_col].sum()
            aggregated_data[f'{app} Total UL (MB)'] = df.groupby('IMSI')[app_ul_col].sum()

        aggregated_data['Total DL (MB)'] = aggregated_data[[f'{app} Total DL (MB)' for app in applications]].sum(axis=1)
        aggregated_data['Total UL (MB)'] = aggregated_data[[f'{app} Total UL (MB)' for app in applications]].sum(axis=1)

        # Create subplots for total download/upload data
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot total download (MB) by application
        total_dl_data = [aggregated_data[f'{app} Total DL (MB)'].sum() for app in applications]
        data_utils.plot_bar_with_annotations(ax1, total_dl_data, applications, 'Total Download Data (MB) by Application', ['skyblue', 'lightgreen', 'lightcoral', 'orange', 'purple', 'gold', 'pink'])

        # Plot total upload (MB) by application
        total_ul_data = [aggregated_data[f'{app} Total UL (MB)'].sum() for app in applications]
        data_utils.plot_bar_with_annotations(ax2, total_ul_data, applications, 'Total Upload Data (MB) by Application', ['skyblue', 'lightgreen', 'lightcoral', 'orange', 'purple', 'gold', 'pink'])

        # Adjust layout and display
        plt.tight_layout()
        st.pyplot(fig)

    elif selection == "EDA":
        st.subheader("Exploratory Data Analysis (EDA)")
        
        # Prepare DataFrame
        df = data_utils.convert_bytes_to_megabytes()
        df['Total Duration (ms)'] = df['Dur. (ms)']
        df['Total Data (MB)'] = df[['Social Media DL (MB)', 'Google DL (MB)', 'Email DL (MB)',
                                    'Youtube DL (MB)', 'Netflix DL (MB)', 'Gaming DL (MB)', 
                                    'Other DL (MB)', 'Social Media UL (MB)', 'Google UL (MB)', 
                                    'Email UL (MB)', 'Youtube UL (MB)', 'Netflix UL (MB)', 
                                    'Gaming UL (MB)', 'Other UL (MB)']].sum(axis=1)
        df['Decile'] = pd.qcut(df['Total Duration (ms)'], 10, labels=False)
        decile_summary = df.groupby('Decile')['Total Data (MB)'].sum().reset_index()

        st.subheader("Decile Summary")
        st.write(decile_summary)

        st.subheader("Total Data by Decile")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(decile_summary['Decile'], decile_summary['Total Data (MB)'], color='skyblue')
        ax.set_xlabel('Decile')
        ax.set_ylabel('Total Data (MB)')
        ax.set_title('Total Data by Decile')
        ax.set_xticks(decile_summary['Decile'])
        ax.set_xticklabels([f'Decile {i}' for i in decile_summary['Decile']])
        plt.tight_layout()
        st.pyplot(fig)

        # Compute and display basic statistics
        basic_metrics = df[['Total DL (MB)', 'Total UL (MB)', 'Total Data (MB)']].agg(['mean', 'median', 'std', 'min', 'max'])
        st.subheader("Basic Metrics")
        st.write(basic_metrics)

        # Compute and display statistical summaries and visualizations
        quantitative_vars = df.select_dtypes(include=[np.number]).columns
        statistics = data_utils.compute_statistics(quantitative_vars)
        st.subheader("Non-Graphical Univariate Analysis")
        st.write(statistics)
        
        st.subheader("Graphical Univariate Analysis")
        variables_to_analyze = ['Social Media DL (MB)', 'Google DL (MB)', 'Email DL (MB)', 
                                'Youtube DL (MB)', 'Netflix DL (MB)', 'Gaming DL (MB)', 
                                'Other DL (MB)']
        fig = data_utils.plot_univariate_analysis(variables_to_analyze)
        st.pyplot(fig)
        
        st.subheader("Graphical Bivariate Analysis")
        total_dl_col = 'Total DL (MB)'
        total_ul_col = 'Total UL (MB)'
        bivariate_results, figs = data_utils.bivariate_analysis(variables_to_analyze, total_dl_col, total_ul_col)

        # Display the correlation results
        st.write("Correlation Table")
        st.dataframe(bivariate_results)  # Display the DataFrame of correlation results

        # Display each plot in Streamlit
        st.write("Scatter Plots with Regression Lines")
        for fig in figs:
            st.image(fig, use_column_width=True)
elif task == "User Engagement Analysis":
    st.subheader("User Engagement Analysis")

    selection = st.sidebar.radio("Select Section", ["Engagement Metrics", "K-Means Clustering", "Cluster Metrics","Aggregated User Traffic"])

    if selection == "Engagement Metrics":
        st.subheader("Customer Engagement Metrics")

        # Analyze and display engagement metrics
        results = data_utils.analyze_customer_engagement()
        for key, value in results.items():
            st.write(f"**{key}:**")
            st.write(value)
    elif selection == "K-Means Clustering":
        st.subheader("K-Means Clustering Analysis")
        df=data_utils.analyze_customer_engagement() 
            # Perform clustering
        df_with_clusters, kmeans_model = data_utils.perform_kmeans_clustering(k=3)
            
            # Display DataFrame with clusters
        #  st.write(df_with_clusters.head())
            
            # Review the cluster centers and sizes
        st.write("Cluster centers:\n", kmeans_model.cluster_centers_)
        st.write("Number of customers in each cluster:\n", df_with_clusters['Cluster'].value_counts())
            
            # Plot clusters
        fig=data_utils.plot_clusters()
        st.pyplot(fig)  # Display the plot in Streamlit
        st.write("""
We can identify the typical characteristics of each segment—such as high versus low data usage—by analyzing the cluster centers. This segmentation enables the development of customized strategies, such as targeted marketing campaigns or specialized support for each cluster. For instance, low-usage clusters might be offered special deals, while high-usage clusters could benefit from premium subscription packages. Understanding these segments also aids in optimizing resources, improving network design and performance in areas with high data consumption.

Additionally, clustering helps in identifying outliers—customers who don't fit neatly into any segment—highlighting those with specific needs or issues that may require additional attention. Overall, clustering provides a clearer picture of customer types and behaviors, supporting more effective strategic planning and decision-making.
""")
    elif selection == "Cluster Metrics":
            st.subheader("Cluster Metrics Analysis")
            df=data_utils.analyze_customer_engagement() 
            df_with_clusters, kmeans_model = data_utils.perform_kmeans_clustering(k=3)

    # Define the metrics to be used for cluster analysis
            metrics = ['Session_duration', 'DL_data', 'UL_data', 'Total DL (MB)', 'Total UL (MB)']
    
    # Compute and display cluster metrics
            cluster_metrics, fig = data_utils.compute_and_plot_cluster_metrics(metrics)  # Pass 'metrics' as an argument
    
    # Display the computed cluster metrics
            st.write(cluster_metrics)
            st.pyplot(fig)
    elif selection == "Aggregated User Traffic":
        st.subheader("Aggregated User Traffic")
        df=data_utils.analyze_customer_engagement() 
        data_utils.aggregate_user_traffic()
        data_utils.top_engaged_users()
        # Plot the top 3 most used applications
        st.write("Top 10 engaged users for each application:")
        for app, top_users in data_utils.top_10_users_per_app.items():
            st.write(f"Top 10 users for {app}:")
            st.write(top_users)  # Display top users for each application
        
        fig= data_utils.plot_top_applications()
        
        st.pyplot(fig)
        st.write("User 33663706799 has indeed exhibited significant data usage across multiple platforms:")        
elif task == "Experience Analytics":
    st.subheader("Experience Analytics")
    
    # User Experience Analysis
    st.subheader("User Experience Metrics")

    experience_metrics = data_utils.analyze_user_experience()
    for key, value in experience_metrics.items():
        st.write(f"**{key}:**")
        st.write(value)

    st.subheader("Experience Metrics Visualization")
    
    # Visualization for Experience Metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    experience_values = [value for value in experience_metrics.values()]
    experience_labels = [key for key in experience_metrics.keys()]
    
    ax.bar(experience_labels, experience_values, color=['lightblue', 'lightgreen', 'lightcoral'])
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title('User Experience Metrics')
    plt.xticks(rotation=45, ha='right')

    for i, value in enumerate(experience_values):
        ax.text(i, value + 0.05 * max(experience_values), f'{value:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    st.pyplot(fig)

elif task == "Experience Analytics":
    st.markdown("## Experience Analytics")
    # Code for Experience Analytics here

elif task == "Satisfaction Analysis":
    st.markdown("## Satisfaction Analysis")
    # Code for Satisfaction Analysis here
