from torch_geometric.nn import MLP


class SimpleMLP(MLP):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        n_layers=1,
        act="silu",
        dropout=0.0,
        norm=None,
        plain_last=False,
        **kwargs,
    ):
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        super().__init__(
            channel_list=None,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=n_layers,
            dropout=dropout,
            act=act,
            act_first=kwargs.get("act_first", False),
            act_kwargs=kwargs.get("act_kwargs", None),
            norm=norm,
            norm_kwargs=kwargs.get("norm_kwargs", None),
            plain_last=plain_last,
            bias=kwargs.get("bias", True),
        )
