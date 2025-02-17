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
        norm="batch_norm",
        plain_last=False,
        **kwargs,
    ):
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        super().__init__(
            channel_list=None,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=n_layers,
            act=act,
            dropout=dropout,
            norm=norm,
            plain_last=plain_last,
            **kwargs,
        )
