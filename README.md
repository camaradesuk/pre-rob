## Preclinical risk of bias assessment with CNN/Attention/HAN/BERT

Predict reporting scores and extract relevant sentences of five risk of bias items for preclinical publications:

- Random Allocation to Treatment/Control Group
- Blinded Assessment Outcome
- Conflict of Interest
- Compliance of Animal Welfare Regulations
- Animal Exclusions

### Usage

#### Clone source code

```
git clone https://github.com/camaradesuk/pre-rob.git
```

---

#### Run the Docker

The Docker image for this project is available on Docker Hub:  
**`karmakarmala/pre-rob-app:latest`**

##### Steps to Use Docker:

1. **Pull the Docker image**:

   ```bash
   docker pull karmakarmala/pre-rob-app:latest
   ```

2. **Set up Input/Output Folders**:
   Create two directories in your working directory:

   - `input`: Place your input file(s) here.
   - `output`: The app will save the results in this folder.

   ```bash
   mkdir input output
   ```

3. **Place Input File(s)**:
   Move your input file (e.g., `example.csv`) to the `input` folder:

   ```bash
   mv your-input-file.csv input/
   ```

4. **Run the Docker Container**:

   ```bash
   docker run --rm -v ${PWD}/input:/input -v ${PWD}/output:/output karmakarmala/pre-rob-app:latest
   ```

   - `--rm`: Automatically removes the container after execution.
   - `-v ${PWD}/input:/input`: Maps your local `input` folder to the container’s `/input` folder.
   - `-v ${PWD}/output:/output`: Maps your local `output` folder to the container’s `/output` folder.

5. **View the Output**:
   Check the `output` folder for results.

---

#### CSV file including txt paths as input

It should have two columns: 'id' and 'path'.

See [input.csv](https://github.com/camaradesuk/pre-rob/blob/robSetup/rob-app/example/input.csv) for example.
The 'path' column stores the absolute paths of TXT files.

```
python rob.py -p ../pre-rob/rob-app/example/input.csv  # absolutae path of input.csv
# Extract two relevant sentences for each item
python rob.py -p ../pre-rob/rob-app/example/input.csv -s 2
```

Results are saved in [output.csv](https://github.com/camaradesuk/pre-rob/blob/robSetup/rob-app/example/example_output.csv).

### Citation

[![DOI](https://zenodo.org/badge/222727172.svg)](https://zenodo.org/badge/latestdoi/222727172)
