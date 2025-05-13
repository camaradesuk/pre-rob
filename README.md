## Preclinical risk of bias assessment with CNN/Attention/HAN/BERT

Predict reporting scores and extract relevant sentences of five risk of bias items for preclinical publications:

- Random Allocation to Treatment/Control Group
- Blinded Assessment Outcome
- Conflict of Interest
- Compliance of Animal Welfare Regulations
- Animal Exclusions

-----

### Table of Contents

- [Prerequisites: Install Docker](https://www.google.com/search?q=%23prerequisites-install-docker)
- [Clone Source Code](https://www.google.com/search?q=%23clone-source-code)
- [Using Docker](https://www.google.com/search?q=%23using-docker)
  - [1. Build the Docker Image](https://www.google.com/search?q=%231-build-the-docker-image)
  - [2. Set up Input/Output Folders](https://www.google.com/search?q=%232-set-up-inputoutput-folders)
  - [3. Place Input File(s)](https://www.google.com/search?q=%233-place-input-files)
  - [4. Run the Docker Container](https://www.google.com/search?q=%234-run-the-docker-container)
  - [5. View the Output](https://www.google.com/search?q=%235-view-the-output)
- [CSV file including txt paths as input](https://www.google.com/search?q=%23csv-file-including-txt-paths-as-input)
  - [Running the script directly (e.g., for development or if not using Docker)](https://www.google.com/search?q=%23running-the-script-directly-eg-for-development-or-if-not-using-docker)
- [Citation](https://www.google.com/search?q=%23citation)

-----

### Prerequisites: Install Docker

Before you begin, you need to have Docker installed on your system. Docker allows you to run applications in isolated environments called containers.

- **Windows:** Download and install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/).
- **macOS:** Download and install [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/).
- **Linux:** Installation methods vary by distribution.
  - For Debian/Ubuntu: `sudo apt-get update && sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin` (Follow the official Docker guide for [Ubuntu](https://docs.docker.com/engine/install/ubuntu/) or [Debian](https://docs.docker.com/engine/install/debian/)).
  - For Fedora: `sudo dnf -y install dnf-plugins-core && sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo && sudo dnf install docker-ce docker-ce-cli containerd.io docker-compose-plugin` (Follow the official Docker guide for [Fedora](https://docs.docker.com/engine/install/fedora/)).
  - For other distributions, please refer to the [official Docker installation guides](https://docs.docker.com/engine/install/).

Ensure the Docker daemon/service is running after installation.

-----

### Clone Source Code

First, clone the repository to get the `Dockerfile` and all necessary project files:

```bash
git clone https://github.com/camaradesuk/pre-rob.git
cd pre-rob
```

-----

### Using Docker

You can run this project by building a Docker image from the provided `Dockerfile` and then running a container from that image.

#### 1\. Build the Docker Image

Navigate to the root directory of the cloned `pre-rob` project (the directory containing the `Dockerfile`). Run the following command to build the Docker image. This may take some time as it downloads dependencies and sets up the environment.

```bash
docker build -t pre-rob-app .
```

- `docker build`: The command to build an image from a Dockerfile.
- `-t pre-rob-app`: Tags the image with the name `pre-rob-app` (you can choose a different name:tag if you prefer).
- `.`: Specifies that the build context (the set of files to send to the Docker daemon) is the current directory.

#### 2\. Set up Input/Output Folders

Create two directories in your current working directory (e.g., the `pre-rob` directory or any other location on your host machine):

- `input`: Place your input file(s) here.
- `output`: The app will save the results in this folder.

<!-- end list -->

```bash
mkdir input output
```

#### 3\. Place Input File(s)

Move your input file (e.g., `your-input-file.csv`) to the `input` folder you just created:

```bash
mv your-input-file.csv input/
```

*(Ensure `your-input-file.csv` is correctly formatted as described in the [CSV file including txt paths as input](https://www.google.com/search?q=%23csv-file-including-txt-paths-as-input) section.)*

#### 4\. Run the Docker Container

Execute the following command to run the application using the Docker image you built:

```bash
docker run --rm -v ${PWD}/input:/input -v ${PWD}/output:/output pre-rob-app
```

- `docker run`: The command to run a new container from an image.
- `--rm`: Automatically removes the container when it exits. This is useful for keeping your system clean.
- `-v ${PWD}/input:/input`: Mounts your local `input` folder (located in the current working directory, `${PWD}`) to the `/input` folder inside the container. The application reads its input from `/input/input.csv` within the container.
- `-v ${PWD}/output:/output`: Mounts your local `output` folder to the `/output` folder inside the container. The application writes its results to `/output/output.csv` within the container.
- `pre-rob-app`: The name of the image you built earlier. If you used a different tag, replace `pre-rob-app` with `your-image-name:your-tag`.

#### 5\. View the Output

After the container finishes execution, check your local `output` folder for the results (e.g., `output.csv`).

-----

### CSV file including txt paths as input

The input CSV file should have two columns: 'id' and 'path'.

See [input.csv](https://github.com/camaradesuk/pre-rob/blob/robSetup/rob-app/example/input.csv) for an example.
The 'path' column stores the **absolute paths of TXT files as they will be accessible from within the container**.

**Important for Docker Usage:** When running with Docker, the paths in your input CSV must be relative to the `/input` directory *inside the container*.
For example, if you place `my_article.txt` directly into your local `input` folder which is then mounted to `/input` in the container, the path in the CSV should be `/input/my_article.txt`.

However, the application's entrypoint (`ENTRYPOINT ["/bin/bash", "-c", "exec python rob.py -p /input/input.csv -o /output/output.csv"]`) is hardcoded to look for a file named `input.csv` directly inside the `/input` directory of the container. This `input.csv` file will then contain paths to other `.txt` files.

If you intend to process text files that are also in the `input` directory (e.g. `input/text1.txt`, `input/text2.txt`), your `input/input.csv` should look like this:

```csv
id,path
1,/input/text1.txt
2,/input/text2.txt
```

#### Running the script directly (e.g., for development or if not using Docker)

If you were to set up the environment locally without Docker and run the script directly, you would use commands like these:

```bash
# Assuming you are in the /pre-rob/rob-app directory and the conda environment is activated
# Use the absolute path to your input.csv
python rob.py -p /path/to/your/pre-rob/rob-app/example/input.csv

# Extract two relevant sentences for each item
python rob.py -p /path/to/your/pre-rob/rob-app/example/input.csv -s 2
```

Results are saved in [output.csv](https://github.com/camaradesuk/pre-rob/blob/robSetup/rob-app/example/example_output.csv).

-----

### Citation

[](https://zenodo.org/badge/latestdoi/222727172)

-----
