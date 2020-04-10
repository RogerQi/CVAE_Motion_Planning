def dispatcher(cfg):
    network_name = cfg.BACKBONE.network
    if network_name == "dropout_lenet":
        from .classification import dropout_lenet
        return dropout_lenet.net
    elif network_name == "mlp":
        from .classification import mlp
        return mlp.net
    elif network_name == "naive_auto_encoder":
        from .auto_encoder import naive_auto_encoder
        return naive_auto_encoder.net
    elif network_name == "naive_vae":
        from .auto_encoder import naive_vae
        return naive_vae.net
    else:
        raise NotImplementedError