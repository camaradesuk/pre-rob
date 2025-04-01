# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /pre-rob/rob-app

# Install necessary tools
RUN apt-get update && apt-get install -y \
    wget \
    bash \
    && apt-get clean

# Install Miniconda and add it to PATH
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:$PATH"

# Install gdown
RUN pip install gdown

# Download pre-trained weights
RUN mkdir -p pth && \
    gdown "https://drive.google.com/uc?id=18YixZQ4otcZWdAMavy5OviR0579kWrCm" -O pth/dsc_w0.pth.tar

# Copy project files
COPY . /pre-rob

# Make the setup script executable
RUN chmod +x /pre-rob/rob-app/setup.sh

# Run the setup script
RUN /bin/bash -c "/pre-rob/rob-app/setup.sh"

# Update PATH to use the conda environment's binaries.
# This makes sure that python (and other commands) run from /opt/conda/envs/rob/bin
ENV PATH="/opt/conda/envs/rob/bin:$PATH"

# Create input/output folders in the container
RUN mkdir -p /input /output

# Activate the environment
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate rob"

# Set entrypoint to run the app
ENTRYPOINT ["/bin/bash", "-c", "exec python rob.py -p /input/input.csv -o /output/output.csv"]