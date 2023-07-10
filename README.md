# WormSwin
Repository for [WormSwin: Instance segmentation of C. elegans using vision transformer](https://doi.org/10.1038/s41598-023-38213-7) as appeared in Nature Scientific Reports 2023.

Example images taken from the paper.

![instance segmented c. elegans](readme_examples/seg_1.png)
![instance segmented c. elegans](readme_examples/seg_2.png)

We used [MMDetection](https://github.com/open-mmlab/mmdetection) as toolbox for our network.
Please follow their [installation instructions](https://mmdetection.readthedocs.io/en/latest/get_started.html).
To manage our Python packages we used [(mini)conda](https://docs.conda.io/en/latest/miniconda.html).
Our environment YAML file to restore the package versions can be found in this repo.
Python version 3.8, PyTorch 1.11 for CUDA 11.3 and mmcv-full 1.5.3 was used for our setup.

We trained on 4 Nvidia Tesla V100-SMX2 32 GB GPUs, 6 cores of an Intel Xeon Gold 6248 CPU @ 2.50 GHz and 100 GB of RAM. With a batch size of four (one image per GPU) and two workers per GPU, training for 36 epochs took ∼ 19 h.

![tracked c. elegans](readme_examples/tracking.gif)

## Datasets
* [WormSwin Datasets](https://doi.org/10.5281/zenodo.7456803) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7456803.svg)](https://doi.org/10.5281/zenodo.7456803)
* [BBBC010](https://bbbc.broadinstitute.org/BBBC010)


## Citation
```
@article{deserno_wormswin_2023,
	title = {{WormSwin}: {Instance} segmentation of {C}. elegans using vision transformer},
	volume = {13},
	copyright = {2023 The Author(s)},
	issn = {2045-2322},
	shorttitle = {{WormSwin}},
	url = {https://www.nature.com/articles/s41598-023-38213-7},
	doi = {10.1038/s41598-023-38213-7},
	language = {en},
	number = {1},
	urldate = {2023-07-10},
	journal = {Scientific Reports},
	author = {Deserno, Maurice and Bozek, Katarzyna},
	month = jul,
	year = {2023},
	note = {Number: 1
Publisher: Nature Publishing Group},
	keywords = {Behavioural methods, Machine learning, Software},
	pages = {11021},
}
```
