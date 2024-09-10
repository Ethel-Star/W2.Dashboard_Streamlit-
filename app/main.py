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
    st.markdown("## Experience Analytics")
    st.write(" In order to complete the Experience Analytics, we must examine device attributes (handset type) and network performance parameters (TCP retransmission, Round Trip Time (RTT), Throughput) in order to gain insights about user experience.")
    selection = st.sidebar.radio("Select Section", ["Aggregating Information Per Customer", "Throughput and TCP Retransmissions by Handset Type"])
    if selection == "Aggregating Information Per Customer":
        st.subheader("Summary Statistics")
        cleaned_df = data_utils.clean_data()
        aggregated_df = data_utils.aggregate_customer_metrics()
        summary_stats = data_utils.summary_statistics(['TCP DL Retrans. Vol (MB)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)'])
        st.write(summary_stats)
        import streamlit as st

        st.write("""
### Network Performance Summary

**1. TCP Downlink Retransmission Volume (MB):**
- **Average:** 0.472 MB
- **Standard Deviation:** 0.206 MB
- **Range:** 0.000002 MB to 1.726 MB

**Key Insight:** The majority of connections show low TCP retransmission volumes, indicating that the network is generally reliable with minimal data loss. However, the presence of some higher values suggests that there are occasional issues with packet loss or network instability. This could impact the quality of streaming or file transfers and might require further investigation or optimization to ensure consistent performance.

**2. Average Round-Trip Time Downlink (RTT DL, ms):**
- **Average:** 41.75 ms
- **Standard Deviation:** 14.00 ms
- **Range:** 0.000 ms to 98.00 ms

**Key Insight:** The average round-trip time is fairly low, which is good news for applications requiring real-time interaction, such as online gaming or video conferencing. However, the wide range, with some values reaching up to 98 ms, indicates that some users might experience noticeable delays. High latency can affect user experience and may need attention to identify and address the sources of delay, such as network congestion or routing issues.

**3. Average Bearer Throughput Downlink (TP DL, kbps):**
- **Average:** 1949.57 kbps
- **Standard Deviation:** 5317.30 kbps
- **Range:** 0.000 kbps to 30523.00 kbps

**Key Insight:** The data shows a significant variation in throughput, with a mean value of nearly 2 Mbps but a large standard deviation. This suggests that while some users benefit from very high speeds, others may experience much lower throughput. This disparity could be due to varying network conditions, user density, or equipment capabilities. Addressing this variability might involve network upgrades, optimizing resource allocation, or improving signal coverage to ensure a more consistent user experience across the network.
""")
        st.subheader("Top 10, bottom 10, and most frequent values for TCP, RTT, and Throughput.")
        fig=data_utils.plot_all_statistics()
        st.pyplot(fig)
    elif selection == "Throughput and TCP Retransmissions by Handset Type":
        st.subheader("Distribution of Average Throughput and TCP Retransmissions by Handset Type")
        throughput_distribution = data_utils.report_throughput_distribution()
        print("Average Throughput Distribution per Handset Type:")
        st.dataframe(throughput_distribution, use_container_width=True)
        st.write("""
        **Performance Disparity:** There is a noticeable disparity in both throughput and TCP retransmissions among different handset types. Higher-end devices generally perform better in terms of throughput and show lower retransmission volumes.
    
        **User Experience:** Users with older or lower-end devices might face slower data speeds and less reliable connections, highlighting potential areas for network and device improvement.
        """)
        data_utils.scale_numeric_data()
        data_utils.apply_kmeans_clustering(n_clusters=3)
        st.subheader("Cluster Centers")
        cluster_centers_df = data_utils.get_cluster_centers()
        st.dataframe(cluster_centers_df, use_container_width=True)
        st.subheader("Cluster Descriptions")

# Description for Cluster 1
        st.write("**Cluster 1 Description:**")
        st.write("Average Throughput (kbps): 190.49")
        st.write("Average TCP Retransmission (MB): 0.52")
        st.write("Average RTT (ms): 40.65")
        st.write("")  # Add a blank line

# Description for Cluster 2
        st.write("**Cluster 2 Description:**")
        st.write("Average Throughput (kbps): 201.95")
        st.write("Average TCP Retransmission (MB): 0.52")
        st.write("Average RTT (ms): 40.93")
        st.write("")  # Add a blank line

# Description for Cluster 3
        st.write("**Cluster 3 Description:**")
        st.write("Average Throughput (kbps): 10007.43")
        st.write("Average TCP Retransmission (MB): 0.26")
        st.write("Average RTT (ms): 46.22")
    # Print cluster descriptions
        st.subheader("Cluster Descriptions")
        cluster_descriptions = data_utils.describe_clusters()
        st.dataframe(cluster_descriptions, use_container_width=True)
        figs = data_utils.visualize_clusters()
        for fig_key, buf in figs.items():
            st.write(fig_key)
            st.image(buf, caption=fig_key)
        st.subheader("Cluster Analysis")

# Analysis of Cluster 1
        st.write("""
**Cluster 1 Analysis:**
The analysis of Cluster 1 reveals moderate network performance. It has an average throughput of **190.49 kbps**, a low TCP retransmission volume of **0.52 MB**, and an RTT of **40.65 ms**. This suggests that users in this cluster experience steady, average-speed connections.
""")

# Analysis of Cluster 2
        st.write("""
**Cluster 2 Analysis:**
Cluster 2 demonstrates slightly improved performance over Cluster 1. It features a higher average throughput of **201.95 kbps**, similar retransmission rates of **0.52 MB**, and an RTT of **40.93 ms**. This indicates that users in this cluster enjoy slightly better network speeds but with comparable overall performance.
""")

# Analysis of Cluster 3
        st.write("""
**Cluster 3 Analysis:**
Cluster 3 stands out significantly with an average throughput of **10,007.43 kbps**, far exceeding the other clusters. It also has a smaller TCP retransmission volume of **0.26 MB**, indicating more efficient data delivery. However, the RTT for this cluster is greater at **46.22 ms**, suggesting a minor trade-off in latency for increased throughput. Overall, Cluster 3 likely represents users with substantially higher bandwidth or better network conditions, while Clusters 1 and 2 reflect more typical user experiences, with Cluster 2 showing a minor improvement in throughput.
""")
elif task == "Satisfaction Analysis":
    st.markdown("## Satisfaction Analysis")
    selection = st.sidebar.radio("Select Section", ["Satisfaction Score", "Regression Model"])
    

# Display your name at the bottom of the sidebar
st.sidebar.markdown("""
<div style='position: absolute; bottom: 0; width: 100%; text-align: center;'>
    <strong>10 Academy: Ethel.C</strong>
</div>
""", unsafe_allow_html=True)