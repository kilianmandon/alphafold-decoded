# AlphaFold Decoded

This project aims to provide an easy-to-understand implementation of AlphaFold, designed for educational purposes. The repository will be structured with a series of Jupyter Notebooks or Python files containing interactive exercises and TODO markers. By completing these coding exercises, users will gain a hands-on understanding of AlphaFold's core concepts.

## Project Description

AlphaFold revolutionized the field of protein structure prediction. This project breaks down AlphaFold's complex processes into manageable steps. Users will actively participate in the implementation process by:

* Filling out missing code sections within provided Jupyter Notebooks or Python files.
* Following the guidance of TODO markers.
* Referencing solution files for assistance and to check their work.

## Folder Structure

* **current_implementation:** Contains the initial work-in-progress code base for the AlphaFold implementation.
* **tutorials:** Will house Jupyter Notebooks or Python files containing interactive exercises designed to guide users through implementing AlphaFold components. These exercises will be populated by reformatting and adapting content from the `current_implementation` folder. 
* **solutions:** Provides solutions to the coding exercises.  

## Setup Instructions

1. **Install Mamba (if needed):**

   * If you don't have Mamba installed, it provides faster dependency resolution than the standard `conda`. Download instructions can be found on the Mambaforge website: https://mamba.readthedocs.io/en/latest/

2. **Install Dependencies**

   * **Choose the right environment file:**
      * **environment_cpu.yml:** Use this if you **don't** have a compatible NVIDIA GPU 
      * **environment_cuda.yml:** Use this if you **do** have a compatible NVIDIA GPU

   * **Use Mamba to install the environment:**
      ```bash
      mamba env create -f environment_cpu.yml  # Or environment_cuda.yml
      ```

3. **Activate the Environment**

   * Before running your notebooks, activate the project's environment:
      ```bash
      conda activate alphafold
      ```

4. **Select the Kernel in Jupyter Notebook**

   * When you open a Jupyter Notebook, ensure the correct kernel is selected. The kernel name should match your environment name (e.g., 'alphafold'). You'll usually find the kernel selection option in the toolbar or a "Kernel" menu within your notebook. 
