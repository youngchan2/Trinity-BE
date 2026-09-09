class ResourceBusy(RuntimeError):
    """A live execution, Graph, Tensor storage or submission prevents this operation."""


class RuntimeFailure(RuntimeError):
    def __init__(
        self,
        message,
        *,
        domain=0,
        code=0,
        stage=0,
        rank=-1,
        submitted=False,
        cleanup_domain=0,
        cleanup_code=0,
    ):
        super().__init__(message)
        self.domain = domain
        self.code = code
        self.stage = stage
        self.rank = rank
        self.submitted = submitted
        self.cleanup_domain = cleanup_domain
        self.cleanup_code = cleanup_code


class DistributedFailure(RuntimeFailure):
    """Ranks disagree or a world has failed; unsafe resources remain pinned."""
