# Use Python slim image for efficiency
FROM python:3.9-slim

# Set the working directory
WORKDIR /pre-rob

# Install necessary tools
RUN apt-get update && apt-get install -y \
    wget \
    bash \
    && apt-get clean

# Install Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Copy the repository code and bash script
COPY . /pre-rob

# Make setup.sh executable
RUN chmod +x /pre-rob/rob-app/setup.sh

# Run the setup script to install dependencies
RUN /bin/bash -c "/pre-rob/rob-app/setup.sh"

# Set entrypoint to activate the environment and run the app
ENTRYPOINT ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate rob && python rob.py $@"]