<div align="center">
<h1><b>TransformerX</b></h1>
<hr>
<p><b>TransformerX</b> is a Python library for building transformer-based models.</p>
</div>

<div align="center">
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/emgraph">
<img alt="PyPI - Implementation" src="https://img.shields.io/pypi/implementation/transformerx">
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/tensorops/transformerx">
<img alt="PyPI - Maintenance" src="https://img.shields.io/badge/Maintained%3F-yes-green.svg">
<img alt="PyPI - License" src="https://img.shields.io/pypi/l/transformerx.svg">
<img alt="PyPI - Format" src="https://img.shields.io/pypi/format/transformerx.svg">
<img alt="Status" src="https://img.shields.io/pypi/status/transformerx.svg">
<img alt="Commits" src="https://badgen.net/github/commits/tensorops/transformerx">
<img alt="Commits" src="https://img.shields.io/badge/TensorFlow 2-FF6F00?style=flat&logo=tensorflow&logoColor=white">
</div>

<div align="center">
<p>It comes with multiple building blocks and layers you need for creating your model.</p>
<hr>
</div>

<div>
    <p>Join <a href="https://discord.gg/WGdPS5NJ"><b>TensorOps</b> community on Discord</a></p>
    <p>Follow <a target="_blank" href="https://twitter.com/tensorops"><b>TensorOps</b> Twitter</a></p>
</div>

<div>
  <h2>Installation</h2>
  <p>Install the latest version of <b>TransformerX</b>:</p>
  <pre>$ pip install transformerx</pre>
</div>

<div>
    <h2>Example</h2>
<p>This is a French to English translation model.</p>
<b>Note</b>: The <code>data_loader</code> and <code>training</code> modules are still under development and you may 
want to use your own training and input pipeline. However, 
the <code>layers</code> package is the core component and will remain the same (you can integrate it with Tensorflow already 🔜 Pytorch and JAX). 

```python
from transformerx.data_loader import BaseDataset
from transformerx.training import Transformer, Trainer
from transformerx.layers import TransformerEncoder, TransformerDecoder

depth, n_blocks, dropout = 256, 2, 0.2
ffn_num_hiddens, num_heads = 64, 4
key_size, query_size, value_size = 256, 256, 256

data = BaseDataset(batch_size=128)
norm_shape = [2]
encoder = TransformerEncoder(
    len(data.src_vocab),
    depth,
    norm_shape,
    ffn_num_hiddens,
    num_heads,
    n_blocks,
    dropout,
)
decoder = TransformerDecoder(
    len(data.tgt_vocab),
    depth,
    norm_shape,
    ffn_num_hiddens,
    num_heads,
    n_blocks,
    dropout,
)
model = Transformer(encoder, decoder, tgt_pad=data.tgt_vocab["<pad>"], lr=0.001)
trainer = Trainer(max_epochs=2, gradient_clip_val=1)
trainer.fit(model, data)
```

</div>

<div>
<h2>Features</h2>

- [x] Support CPU/GPU
- [x] Vectorized operations
- [x] Standard API

</div>
<h2>If you found it helpful, please give us a <span>:star:</span></h2>

<div>
<h2>License</h2>
<p>Released under the Apache 2.0 license</p>
</div>

<div class="footer"><pre>Copyright &copy; 2021-2022 <b>TensorOps</b> Developers

<a href="https://soran-ghaderi.github.io/">Soran Ghaderi</a> (soran.gdr.cs@gmail.com)
follow me
on <a href="https://github.com/soran-ghaderi"><img alt="Github" src="https://img.shields.io/badge/GitHub-100000?&logo=github&logoColor=white"></a> <a href="https://twitter.com/soranghadri"><img alt="Twitter" src="https://img.shields.io/badge/Twitter-1DA1F2?&logo=twitter&logoColor=white"></a> <a href="https://www.linkedin.com/in/soran-ghaderi/"><img alt="Linkedin" src="https://img.shields.io/badge/LinkedIn-0077B5?&logo=linkedin&logoColor=white"></a>
<br>
<a href="https://uk.linkedin.com/in/taleb-zarhesh">Taleb Zarhesh</a> (taleb.zarhesh@gmail.com)
follow me
on <a href="https://github.com/sigma1326"><img alt="Github" src="https://img.shields.io/badge/GitHub-100000?&logo=github&logoColor=white"></a> <a href="https://twitter.com/taleb__z"><img alt="Twitter" src="https://img.shields.io/badge/Twitter-1DA1F2?&logo=twitter&logoColor=white"></a> <a href="https://www.linkedin.com/in/taleb-zarhesh/"><img alt="Linkedin" src="https://img.shields.io/badge/LinkedIn-0077B5?&logo=linkedin&logoColor=white"></a>
</pre>
</div>
