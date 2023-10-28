# Parts-aware-DINO
Official implementation of Parts based Attention for Highly Occluded Pedestrian Detection with Transformers

## Installation

<details>
  <summary>Installation</summary>
  
  We use the similar instructions as mentioned in the 
  [base repository](https://github.com/IDEA-Research/DINO).

   1. Clone this repo
   ```sh
   git clone https://github.com/ajayshastry08/Flow-Based-Matcher
   cd Flow-Based-Matcher
   ```

   2. Install Pytorch and all other required packages

   You can directly create a conda environment with all required packages by running the script
   ```
   sh install_environment.sh
   ```

   3. Compiling CUDA operators
   ```sh
   cd models/dino/ops
   python setup.py build install
   # unit test (should see all checking is True)
   python test.py
   cd ../../..
   ```
</details> 

### Datasets Preparation and Evaluation scripts
Please refer to [Pedestron repository](https://github.com/hasanirtiza/Pedestron) for dataset preparation and evaluation scripts.

# Benchmarking 
### Benchmarking of our network on pedestrian detection datasets
| Dataset            | &#8595;Reasonable |  &#8595;Small   |  &#8595;Heavy   | 
|--------------------|:----------:|:--------:|:--------:|
| EuroCityPersons        |  **3.7**   | **10.4** | **19.9** |  
| TJU-Pedestrian-Traffic        |  **17.4**   | **24.7** | **52.68** |  
| TJU-Pedestrian-Campus        |  **21.83**   | **37.04** | **57.08** |  
| CityPersons        |  **8.3**   | **15.56** | **27.07** |  
| Caltech Pedestrian |  **2.0**   | **2.8**  | **38.6** |

## Training Details
* To ensure the correct directory is being used for training and validation data, adjustments need to be made in the [file](datasets/coco.py).
* The configuration file used for training can be found at [config file](config/DINO/DINO_4scale_swin.py). Citypersons dataset was trained using a batch size of 2, while Caltech was trained using a batch size of 4.
* During training, it is important to adjust the maximum image size as needed. Details can be seen in the [transform file](config/DINO/coco_transformer.py)

* The following command can be used for multi GPU training.
  ```shell 
  python3 -m torch.distributed.launch --nproc_per_node= gpu_count --master_port=11001 main.py --output_dir path/to/output -c path/to/config --coco_path /path/to/dataset/ --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 backbone_dir=/path/to/backbone/ --pretrain_model_path /path/to/pretrain_model --initilize_cross_attention --finetune_ignore label_enc.weight class_embed 
  ```
* Pretrained models can be found in the [link](https://csciitd-my.sharepoint.com/:f:/g/personal/anz197518_iitd_ac_in/EgAsiTF6lxZPurvohkTsxCAB8IbM6tC-2fEgSUcVVtI2Qg?e=hzBeyu)

# References

* [Pedestron](https://openaccess.thecvf.com/content/CVPR2021/papers/Hasan_Generalizable_Pedestrian_Detection_The_Elephant_in_the_Room_CVPR_2021_paper.pdf)
* [DINO](https://arxiv.org/pdf/2203.03605.pdf)
* [Align-DETR](https://arxiv.org/abs/2304.07527)