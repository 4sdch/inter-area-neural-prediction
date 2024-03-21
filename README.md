# **Inter-Area Neural Prediction**

This repository contains code for analyzing inter-area neural communication via Ridge Regression within the visual cortex by using one area to predict the other). Each Jupyter notebook focuses on a specific aspect of the analysis and can be run independently. Inter-area prediction is done across different areas (V1 predicting V4 in macaque or cortical layer 4 (L4) predicting layer 2/3 (L2/3) in moise) across different stimulus types and recording techniques and in the absence of any stimuli. 

## **Data Acquisition**

The code assumes users will download the data required for the analysis from the two open sources:

[Mouse Dataset](https://figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_in_response_to_2_800_natural_images/6845348?file=12462734)

[Macaque Dataset](https://gin.g-node.org/NIN/V1_V4_1024_electrode_resting_state_data)


## **Dependencies**

To run this project locally, you'll need to install the necessary dependencies. These dependencies are listed in the `requirements.txt` file. Here's how to install them:

1. **Python Version:** Make sure you have Python 3.x installed. You can check by running `python --version` in your terminal.

2. **Install Dependencies:** Navigate to the project directory and install the required libraries using the following command:
   ```bash
   pip install -r requirements.txt


## **Running the Notebooks**

1. Clone this [repository](https://github.com/4sdch/inter-area-neural-prediction)
2. Open a terminal or command prompt and navigate to the project directory.
3. Each Jupyter notebook can be run independently using the following command (replace `<notebook_name>` with the actual filename):

jupyter notebook code/<notebook_name>.ipynb

This will open the notebook in your browser where you can see the code execution and results.

## Code Structure

* **code:** This folder contains the Jupyter notebooks for each figure in the paper.
* **utils:** This folder contains helper functions used across the notebooks (data processing, plotting, etc.).

## Notebooks

* **figure_2_regressions.ipynb:** This notebook performs a regression analysis to predict neural activity of one area using the activity of the other area in mouse and macaue. Shows sample activity along with predictability distributins in mice and macaque.
* **figure_3_directionality.ipynb:** This notebook analyzes the directionality of inter-areal predictability. Is L4 better at predicting L2/3 than vice versa?
* **figure_4_stimulus_types.ipynb:**  This notebook investigates inter-area prediction across different stimulus types like orientation gratings, natural images, checkerboard images, moving bars.
* **figure_5_spont_comparisons.ipynb:** This notebook compares inter-areal prediction in the presence vs. absence of any stimulus.
* **figure_6_neuron_properties.ipynb:** This notebook investigates different properties influencing what makes a neuron/neural site predictable. Properties include signal-to-noise ratio, split-half reliability, receptive field overlap, etc.
* **figure_7_predictability_across_time.ipynb:** This analysis inter-areal prediction across time.
