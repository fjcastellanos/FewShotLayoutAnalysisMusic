# A Few-shot Neural Approach for Layout Analysis of Music Score Images

This is a repository with the source code used in the paper titled "A Few-shot Neural Approach for Layout Analysis of Music Score Images".
This work was published in the International Society for Music Information Retrieval Conference (ISMIR), Milan (Italy), 2023.

Please, cite the following work for any use of the code or for a reference to the pusblished work:

```
@inproceedings{Castellanos23_Ismir,
  author    = {Francisco J. Castellanos and
               Antonio Javier Gallego and
               Ichiro Fujinaga},
  title     = {A Few-shot Neural Approach for Layout Analysis of Music Score Images},
  booktitle = {Proceedings of the 24th International Society for Music Information
               Retrieval Conference, Milan, Italy, November 5-9, 2023},
  pages     = {106--113},
  year      = {2023}
  }
```

# Installation Dependencies

## Python dependencies:

  * h5py (2.10.0)
  * Keras (2.2.4)
  * numpy (1.16.0)
  * scipy (1.2.2)
  * Tensorflow (2.4)
  * opencv-python (4.2.0.32)


## Keras configuration

The code needs *Keras* and *TensorFlow* to be installed. It can be easily done through **pip**. 

*Keras* works over both Theano and Tensorflow, so after the installation check **~/.keras/keras.json** so that it looks like:

~~~
{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
~~~

# How to use
The source code is a python code with the main function in "main.py".
It accepts several parameters to configure the experiments:
  * **-db_train_src** `Path to the folder with the training images.` (**Example:** *datasets/training/images*)
  * **-db_train_gt** `Path to the folder with the training ground-truth images.` (**Example:** *datasets/training/gt*)
  * **-db_test_src** `Path to the folder with the testing images.` (**Example:** *datasets/test/images*)
  * **-db_test_gt** `Path to the folder with the testing ground-truth images.` (**Example:** *datasets/test/gt*)
  * **-aug** `List of data augmentation types to be used for training the model. Possible modes are: "none", "random", "flipH", "flipV", "rot", "scale". `(**Example:** *random* (to use random augmentations for all the types)
  * **-window_w** `Window width for the samples extracted from the images. (**Example:** *256*)
  * **-window_h** `Window height for the samples extracted from the images. (**Example:** *256*)
  * **-l** `Depth of the encoder/decoder of the neural network. (**Example:** *4*)
  * **-f** `Number of filters for the architecture. (**Example:** *16*)
  * **-k** `Kernel size. (**Example:** *3*)
  * **-drop** `Dropout rate. (**Example:** *0.4*)
  * **-pages_train** `Number of pages to be used in training. Value -1 uses all the images. (**Example:** *4*)
  * **-npatches** `Number of patches extracted from each page. Value -1 uses all the patches. For random augmentation (proposal), use a number different to -1. (**Example:** *128*)
  * **-n_annotated_patches** `Number of patches annotated within the image. This simulates partial annotations of the images. Value -1 stands for annotating the entire images. (**Example:** *4*)
  * **-e** `Maximum number of epochs to train the model. (**Example:** *200*)
  * **-b** `Batch size. (**Example:** *32*)
  * **-verbose** `To show more details. (**Example:** *1*)
  * **-gpu** `Index of the GPU to be used. (**Example:** *0*)
  * **--test** `Optional parameter. It indicates the test mode, and no training is done. (**Example:** *-test*)
  * **-res** `Optional parameter to be used only in test mode. File to append the results. If the file is not empty, the new results are appended without erosing the content. (**Example:** *results/out.txt*)


Example of use:

~~~
  python -u main.py  
            -db_train_src datasets/b-59-850/training/images 
            -db_train_gt datasets/b-59-850/training/layers/text 
            -db_test_src datasets/b-59-850/test/images
            -db_test_gt datasets/b-59-850/test/layers/text  
            -aug random  
            -window_w 256  
            -window_h 256 
            -l 4  
            -f 32  
            -k 3  
            -drop 0.4  
            -pages_train 3  
            -npatches 128  
            -n_annotated_patches -1  
            -e 200  
            -b 32
            -verbose 1
            -gpu 0
            -b 32
~~~

After training, it should be included the parameter **--test** for testing the model and **-res results/out.txt** for dumping the results in a file. This is optional, since the results are shown in the console.


# Directories
The datasets folder should be in the following structure:
- **datasets**
  - **My dataset**
    - **training**
      - **images** 
        - image_01.png
        - example_99.png
      - **layers**
        - **background**
          - image_01.png
          - example_99.png
        - **staff**
          - image_01.png
          - example_99.png
        - **neume**
          - image_01.png
          - example_99.png
 
    - **test**
      Same structure that the training folder.


The models are saved in the folder "models" automatically, and after testing, the resulting images are saved in folder "test" within the parent folder (not the datasets folder). 

## Acknowledgments

This work was supported by the I+D+i project TED2021-132103A-I00 (DOREMI), funded by MCIN/AEI/10.13039/501100011033. 
This work also draws on research supported by the Social Sciences and Humanities Research Council (895-2013-1012) and the Fonds de recherche du Québec-Société et Culture (2022-SE3-303927).

<a href="https://www.ciencia.gob.es/" target="_blank"><img src="https://www.dlsi.ua.es/~jgallego/projects/DOReMI/images/logo_min.png" style="height:100px" alt="Ministerio de Ciencia e Innovación"></a> 
&nbsp;
<a href="https://commission.europa.eu/strategy-and-policy/recovery-plan-europe_es" target="_blank"><img src="https://www.dlsi.ua.es/~jgallego/projects/DOReMI/images/logo_ue.png" style="height:100px" alt="Financiado por la Unión Europea, fondos NextGenerationEU"></a>
<br>
<a href="https://planderecuperacion.gob.es/" target="_blank"><img src="https://www.dlsi.ua.es/~jgallego/projects/DOReMI/images/logo_plan_recuperacion_transformacion_resiliencia.png" style="height:100px" alt="Plan de Recuperación, Transformación y Resiliencia"></a>
&nbsp;
<a href="https://www.aei.gob.es/" target="_blank"><img src="https://www.dlsi.ua.es/~jgallego/projects/DOReMI/images/logo_aei.png" style="height:100px" alt="Agencia Estatal de Investigación"></a>

<br/>
