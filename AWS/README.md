# AWS

There are two main AWS pipelines that follow similar structure:

![Diagram displaying the cloud architecture.](../pictures/Cloud%20Architecture.png)

## Training Branch

The training branch starts with the base station uploading a CSV file of all the input features of the system throughout the lifetime of the experiment through an API Gateway. This triggers an [AWS Lambda](./upload_training_data.py) that processes the CSV and uploads it to an S3 bucket. The EC2 instance, holding the RL models, then [downloads the CSV file](./download_csv.py) associated with the newest experiment. The RL models are then able to learn based on the input data.

## Deployment Branch

In the deployment branch, multiple CSV files of the outputs are uploaded to a very similar API Gateway based on which model was deployed to the drones. These CSVs get processed by a similar [Lambda](./upload_output_data.py) and are uploaded to a different S3 bucket. On a visualization EC2 instance, a [Prometheus exporter](./Visualization/s3_exporter.py) runs to query the S3 bucket and generate the metrics. A Prometheus server also runs that gets the CSV logs from the exporter. Lastly, a Grafana server is run to visualize the metrics across different logs. All of the processes running on the EC2 instance act as services and start up on boot. Prometheus and Grafana are completely public to view given that the EC2 instance is up, and the IP address of the instance is known.
