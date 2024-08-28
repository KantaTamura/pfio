try:
    import torch
    from pytorch_pfn_extras.profiler import record as ppe_record
    from pytorch_pfn_extras.profiler import \
        record_iterable as ppe_record_iterable

    def record(tag, trace, *args):
        return ppe_record(
            tag,
            use_cuda=torch.cuda.is_available(),
            enable=trace,
            trace=trace,
        )

    def record_iterable(tag, iter, trace, *args):
        return ppe_record_iterable(
            tag,
            iter,
            use_cuda=torch.cuda.is_available(),
            enable=trace,
            trace=trace,
        )

except ImportError:

    class _DummyRecord:
        def __init__(self):
            pass

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    # IF PPE is not available, wrap with noop
    def record(tag, trace, *args):  # type: ignore # NOQA
        return _DummyRecord()

    def record_iterable(tag, iter, trace, *args):   # type: ignore # NOQA
        yield from iter
