# Binary-Label-3D-Shape
## Dictionary
```
.
├── data # The data is not publicly available. The data format is .ply.
│   ├── 3D-CT PLY # Dataset
│   ├── templates # Hierachical template
│   └── put-data-dir-here
├── src # Source code
│   ├── common
│   ├── conf 
│   ├── data 
│   ├── energy 
│   ├── feature 
│   ├── hydra_plugins 
│   ├── mesh 
│   ├── metrics 
│   ├── models 
│   ├── notebooks 
│   ├── train_scripts 
│   ├── visualize 
│   ├── create_dataset_sample_img.py
│   ├── create_feature.py
│   ├── create_latent_distribution.py
│   ├── create_morphing_animation.py
│   ├── create_two_layer_morphing_animation.py
│   ├── evaluate_reconstruction.py
│   ├── latent_variable_visualization.py
│   ├── morphing.py
│   ├── post_process.py
│   ├── post_process_grid_image.py
│   ├── subdivide.py
│   ├── train_mesh_PCA.py
│   ├── train_mesh_VAE.py
│   └── two_layers_hierarchical_morphing.py
├── README.md
├── requirements.txt
└── setup.cfg
```

The following topics will be explained:

1. Usage environment and environment setup
2. Dataset preparation
3. How to run the program
4. Model training
5. Post-processing of training results
6. Morphing

## Environment
Windows Subsystem for Linux 2 (WSL)  
Ubuntu 20.04.6  
cuda 11.8  
Python 3.9.16  
pyenv 2.3.17-9-g528d10e9 (to create virtual environment)  
direnv v2.33.0  

In this project, we use ```hydra``` instead of Python's standard library ```argparse``` for managing program arguments. ```hydra``` reads configuration files written in ```yaml``` notation (located in ```src/conf``` in this project) and makes them available to the program. In addition, files and images output by programs executed with ```hydra``` are saved in a directory automatically created each time the program is executed, so there is no risk of accidentally overwriting results. ```hydra``` supports plugins, which are located in the ```src/hydra_plugins``` directory in this project. Specifically, when debug_mode=False is selected, the program will not execute unless ```git add``` and ```git commit``` are successfully completed each time. This allows for automatic logging of experiments on GitHub, enabling reproducible experiments.
