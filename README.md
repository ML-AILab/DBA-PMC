# DBA-PMC: A Mutually Enhancing Dual-branch Architecture for Pathologic Myopia And Myopic Maculopathy Classification
Chucheng Chen, Zheng Gong, Zhuo Deng, Weihao Gao, Fang Li, **Lan Ma**, Lei Shao, Ruiheng Zhang, Wenbin Wei

*The first two authors contribute equally to this work*

# News

* **2024.05.20** Code and models have been released. :rainbow:
* **2024.02.10** Our paper has been accepted by ISBI2024. :triangular_flag_on_post:

---

**Abstract:** Pathologic myopia is one of the common eye diseases, which is becoming more severe with the increase of myopia prevailing around the world. The diagnosis of pathologic myopia and its co-existing myopic maculopathy is crucial but usually not timely enough due to the lack of experienced ophthalmologists. Therefore, we propose a new dual-branch architecture to detect pathologic myopia and classify myopic maculopathy named DBA-PMC. This architecture can be applied with any feature extraction backbone and can give pathologic myopia prediction and maculopathy classification at the same time. Because pathologic myopia and myopic maculopathy
labels are corelative, two branches of the DBA-PMC can mutually promote the performance of each other through comprehension of this correlation. With extensive experiments, the DBA-PMC surpasses baseline methods with Acc 99.12% for pathologic myopia prediction and mAP 86.095 % for maculopathy classification. This work can help screen and diagnose pathologic myopia and alleviate the work of ophthalmologists.

---

# 1.Create Environment:
 * Python3 (Recommend to use [Anaconda](https://www.anaconda.com/))
 * NVIDIA GPU + CUDA
 * Python packages:
   ```
   cd /DBA-PMC/
   pip install -r requirements.txt
   ```

# 2.Evaluation
(1)Download the pretrained model from([Baidu Disk](https://pan.baidu.com/s/1aOio5sadSImCP4o4111mAw),code:pmc1) and place it to `/DBA-PMC/checkpoints/`.

(2)To test trained model, run
```
cd /DBA-PMC/
CUDA_VISIBLE_DEVICES=0 python test_image_ml.py --modelpath ./checkpoints/model_best_map.pth
```

# 3.Citation
If this repo helps you, please consider citing our work:

```
@inproceedings{DBA-PMC2024,
  title={DBA-PMC: A Mutually Enhancing Dual-branch Architecture for Pathologic Myopia And Myopic Maculopathy Classification},
  author={Chucheng, Chen and Zheng, Gong and Zhuo, Deng and Weihao, Gao and Fang, Li and Lan, Ma and Lei, Shao and Ruiheng, Zhang and Wenbin, Wei},
  booktitle={2024 IEEE 20th International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2024},
  organization={IEEE}
}
```
If you have any questions, please contact me at [malan_ailab@163.com]().
