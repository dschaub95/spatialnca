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
        plain_last: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        """Light-weight wrapper around :class:`torch_geometric.nn.MLP` that
        exposes the most commonly used arguments while ensuring that the
        *bias* flag is honoured. The previous implementation silently ignored
        the provided *bias* argument and always defaulted to ``True`` unless it
        was passed via ``**kwargs`` – a hard-to-spot logic bug that made it
        impossible to disable biases from the call-site.

        Parameters
        ----------
        bias
            Whether to add learnable bias parameters in the linear layers.
        """

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
            bias=bias,
        )
