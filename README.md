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

1. Install a Package Manager (if needed):

* **Conda:** If you already have Conda installed, it will work perfectly fine to manage the project's dependencies. 
* **Mamba:** If you're starting from scratch with environment setup, consider installing Mamba. It's a faster alternative to Conda for dependency resolution and package management.  You can find download instructions on the Mambaforge website: [https://mamba.readthedocs.io/en/latest/](https://mamba.readthedocs.io/en/latest/)

2. Install Dependencies

* **Choose the right environment file:**
   * **environment_cpu.yml:** Use this if you **don't** have a compatible NVIDIA GPU 
   * **environment_cuda.yml:** Use this if you **do** have a compatible NVIDIA GPU
   * **environment_mac.yml:** Use this if you are on a Mac system.

* **Use your chosen package manager to install the environment:**

   ```bash
   # If you've chosen Conda:
   conda env create -f environment_cpu.yml  # Or environment_cuda.yml, or environment_mac.yml

   # If you've chosen Mamba:
   mamba env create -f environment_cpu.yml  # Or environment_cuda.yml, or environment_mac.yml
   ```
      

3. **Activate the Environment**

   * Before running your notebooks, activate the project's environment:
      ```bash
      conda activate alphafold
      ```

4. **Select the Kernel in Jupyter Notebook**

   * When you open a Jupyter Notebook, ensure the correct kernel is selected. The kernel name should match your environment name (e.g., 'alphafold'). You'll usually find the kernel selection option in the toolbar or a "Kernel" menu within your notebook. 

## Working on the Tutorials
All of the tutorials are still being improved. 
The tutorials should be completed in the following order:

1. Tensor Introduction
2. Machine Learning Introduction
3. Attention
4. Feature Extraction
5. Evoformer (work in progress)
6. Geometry (work in progress)
7. Structure Module (work in progress)

If you are already familiar with tensors and machine learning, feel free to start at Attention.

For working on the tutorials, the tutorials folder needs to be your root folder. You can either directly open tutorials in your editor, or you can set "Jupyter: Notebook File Root" as "tutorials" in your workspace settings.