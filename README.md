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
