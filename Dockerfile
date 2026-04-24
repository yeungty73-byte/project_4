FROM uzairakbar/deepracer:v0

# Fix: copy corrected model_metadata.json into the Docker image
# The original baked-in metadata may have parsing issues
COPY custom_files/agent/model_metadata.json /opt/ml/model/model_metadata.json
COPY custom_files/racecar/model_metadata.json /opt/amazon/install/sagemaker_rl_agent/lib/python3.6/site-packages/markov/presets/model_metadata.json
