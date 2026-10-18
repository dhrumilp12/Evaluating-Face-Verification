# Vendored MagFace architecture

`magface_iresnet.py` is an unmodified copy of `models/iresnet.py` from
[IrvingMeng/MagFace](https://github.com/IrvingMeng/MagFace/blob/99bae614ac2643b9694bf18e0c5645272ff6acfa/models/iresnet.py),
commit `99bae614ac2643b9694bf18e0c5645272ff6acfa`.
Its SHA-256 is `9b9670b2190f78782be119a21375e78a81b5444a193ff67d09ca2d48a6b0bc4b`.
The upstream Apache 2.0 license is included as `MagFace-LICENSE`.
The iResNet-100 architecture is instantiated without pretrained loading;
`magface_embedding.py` separately loads the author checkpoint with
`torch.load(weights_only=True)` and requires every backbone state entry.
