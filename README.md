<div align="center">
<h1><b>TransformerX</b></h1>
<hr>
<div>
<img src="logo.jpeg" height="400" width="400">
</div>
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

<div align="center">
    <p>Join <a href="https://discord.gg/WGdPS5NJ"><b>TensorOps</b> community on Discord</a></p>
    <p>Follow <a target="_blank" href="https://twitter.com/tensorops"><b>TensorOps</b> Twitter</a></p>
</div>

<b>Here's why:</b>

- Your time should be focused on creating more advanced and complex tasks
- You shouldn't be doing the same tasks over and over

<div>
  <h2>Installation</h2>
  <p>Install the latest version of <b>TransformerX</b>:</p>
  <pre>$ pip install transformerx</pre>
</div>

<div>
<h2>Getting Started</h2>
<p>This example implements a French to English translation model.</p>

<p>Check out these blog posts on Medium:

- <a href="https://towardsdatascience.com/the-map-of-transformers-e14952226398">The Map Of Transformers</a></p>
- <a href="https://towardsdatascience.com/transformers-in-action-attention-is-all-you-need-ac10338a023a">Transformers in
  Action: Attention Is All You Need</a></p>
- <a href="https://towardsdatascience.com/rethinking-thinking-how-do-attention-mechanisms-actually-work-a6f67d313f99">
  Rethinking Thinking: How Do Attention Mechanisms Actually Work?</a></p>

<b>Note</b>: The <code>data_loader</code> and <code>training</code> modules are still under development and you may
want to use your own training and input pipeline. However,
the <code>layers</code> package is the core component and will remain the same (you can integrate it with Tensorflow
already 🔜 Pytorch and JAX).

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

<div>
<h2>Roadmap</h2>

- [x] Support Tensorflow
- [ ] Extensive documentation
    - <a href="https://github.com/tensorops/TransformerX/issues/30">Documentation</a>
    - Examples (Colab, etc.)
    - Add more contents to the <a href="https://github.com/tensorops/TransformerX/blob/master/README.md">README.md</a>
- [ ] Support Pytorch and JAX
- [ ] More layers
    - <a href="https://github.com/tensorops/TransformerX/issues/44">Attention layers</a>
    - <a href="https://github.com/tensorops/TransformerX/issues/42">Residual and residual gate layers</a>
    - <a href="https://github.com/tensorops/TransformerX/issues/41">Embedding layers</a>

</div>
<div>
<h2>Contribution</h2>

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any
contributions you make are <b>greatly appreciated</b>.

In case you want to get ideas or just work on a ready-to-solve issue, please check out issues with the
label `issue list`.
Here is a <a href="https://github.com/tensorops/TransformerX/issues?q=is%3Aissue+is%3Aopen+label%3A%22issue+list%22">
list of issue lists</a>

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also
simply open an issue with the tag "enhancement" or "bug".

<ol>
<li>Fork the Project</li>
<li>Create your Feature Branch (git checkout -b feature/AmazingFeature)</li>
<li>Commit your Changes (git commit -m 'Add some AmazingFeature')</li>
<li>Push to the Branch (git push origin feature/AmazingFeature)</li>
<li>Open a Pull Request</li>
</ol>

<h3>If you find this project helpful or valuable, please consider giving it a star ⭐️</h3> Your support helps us to
reach
a wider audience and improve the project for everyone. <b>Thank you for your support!</b>
</div>

<div>
<h2>Contributors</h2>
<a href = "https://github.com/Tanu-N-Prabhu/Python/graphs/contributors">
  <img src = "https://contrib.rocks/image?repo=tensorops/transformerx"/>
</a>
</div>

<div>
<h2>License</h2>
<p>Released under the Apache 2.0 license -> To be changed to MIT license after the first stable release</p>
</div>

<h2>Contact</h2>
<div class="footer"><pre>Copyright &copy; 2021-2023 <b>TensorOps</b> Developers
<a href="https://soran-ghaderi.github.io/"><b>Soran Ghaderi</b></a> (soran.gdr.cs@gmail.com)
follow me on <a href="https://github.com/soran-ghaderi"><img alt="Github" src="https://img.shields.io/badge/GitHub-100000?&logo=github&logoColor=white"></a> <a href="https://twitter.com/soranghadri"><img alt="Twitter" src="https://img.shields.io/badge/Twitter-1DA1F2?&logo=twitter&logoColor=white"></a> <a href="https://www.linkedin.com/in/soran-ghaderi/"><img alt="Linkedin" src="https://img.shields.io/badge/LinkedIn-0077B5?&logo=linkedin&logoColor=white"></a>
<br>
<a href="https://uk.linkedin.com/in/taleb-zarhesh"><b>Taleb Zarhesh</b></a> (taleb.zarhesh@gmail.com)
follow me on <a href="https://github.com/sigma1326"><img alt="Github" src="https://img.shields.io/badge/GitHub-100000?&logo=github&logoColor=white"></a> <a href="https://twitter.com/taleb__z"><img alt="Twitter" src="https://img.shields.io/badge/Twitter-1DA1F2?&logo=twitter&logoColor=white"></a> <a href="https://www.linkedin.com/in/taleb-zarhesh/"><img alt="Linkedin" src="https://img.shields.io/badge/LinkedIn-0077B5?&logo=linkedin&logoColor=white"></a>
</pre>
</div>
