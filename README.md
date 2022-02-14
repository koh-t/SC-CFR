# SC-CFR

Codes and datasets for reproducing experiments of 

**<a href="https://arxiv.org/abs/2102.03980">Grab the Reins of Crowds: Estimating the Effects of Crowd Movement Guidance Using Causal Inference</a>**
<br>
<a href="https://koh-t.github.io">Koh Takeuchi</a>,
<a href="https://sites.google.com/view/ryonishida">Ryo Nishida</a>,
<a href="https://hkashima.github.io/index_e.html">Hisashi Kashima</a>,
<a href="http://onishi-lab.jp/index-e.html">Masaki Onishi</a>,
<br>
Presented at [AAMAS 2021](https://aamas2021.soton.ac.uk)

[<a href="http://www.ifaamas.org/Proceedings/aamas2021/pdfs/p1290.pdf">PDF</a>][[Slide](AAMAS.pdf)]

If you find this code useful in your research then please cite
```
@inproceedings{takeuchi2021grab,
  title={Grab the Reins of Crowds: Estimating the Effects of Crowd Movement Guidance Using Causal Inference},
  author={Takeuchi, Koh and Nishida, Ryo and Kashima, Hisashi and Onishi, Masaki},
  booktitle={Proceedings of the 20th International Conference on Autonomous Agents and MultiAgent Systems},
  pages={1290--1298},
  year={2021}
}
```

### Requirements
* python 3
* To install requirements:

```setup
conda env create -f environment.yml
conda activate simon_env
```

### Main analysis
* see `./script/` for commands for running experiments.
* Further details are documented within the code.