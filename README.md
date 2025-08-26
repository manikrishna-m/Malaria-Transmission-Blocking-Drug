The GitHub repository titled **Malaria-Transmission-Blocking-Drug** by [manikrishna-m](https://github.com/manikrishna-m/Malaria-Transmission-Blocking-Drug) presents a data processing and visualization pipeline designed to support research in malaria transmission-blocking drug development.

---

### 🔬 Project Overview

This repository offers a structured pipeline utilizing the PHIDDLI model to process and analyze malaria-related datasets. The pipeline is tailored for handling raw image data, converting it into a format suitable for training and evaluating machine learning models aimed at identifying and understanding malaria transmission stages.([GitHub][1])

---

### ⚙️ Key Features

* **Data Conversion**: Transforms raw image data into standardized JPG format for consistency.
* **Label Extraction**: Employs YOLOv5 to generate bounding boxes, facilitating precise localization of malaria-infected cells within images.
* **Cell Segmentation**: Isolates individual cells from images, enabling detailed analysis of each cell's characteristics.
* **Data Organization**: Structures processed data into directories (`data/images/`, `data/labels/`, `data/cells/`) for efficient access and management.([GitHub][2])

---

### 📁 Repository Structure

* `scripts/`: Contains Python scripts for data processing tasks.
* `data/`: Organized folders for raw data (`src/`), processed images (`images/`), labels (`labels/`), and segmented cells (`cells/`).
* `requirements.txt`: Lists Python dependencies required to run the scripts.
* `dvc.yaml` & `dvc.lock`: Configuration files for Data Version Control, ensuring reproducibility and versioning of datasets.

---

### 🛠️ Installation & Usage

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/manikrishna-m/Malaria-Transmission-Blocking-Drug.git
   cd Malaria-Transmission-Blocking-Drug
   ```



2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```



3. **Prepare Data**:
   Download the malaria dataset from [BioImages](https://www.ebi.ac.uk) and place it in the `data/src/` directory.

4. **Run Data Processing Scripts**:
   Execute the following commands to process the data:

   ```bash
   python scripts/convert_raw.py --input-folder data/src/ --output-folder data/images/ --output-format jpg
   python scripts/extract_yolov5_bboxes.py --input-folder data/src/ --output-folder data/labels/
   python scripts/extract_cells.py --labels-folder data/labels/ --images-folder data/images/ --output-folder data/cells/
   ```



---

### 📌 Notes

* **Data Structure**: Ensure the dataset is organized as specified to maintain compatibility with the pipeline.
* **Model Integration**: While the repository focuses on data preprocessing, the structured output facilitates integration with machine learning models for further analysis.([GitHub][3])

---

