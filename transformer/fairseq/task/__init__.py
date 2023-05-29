import argparse
import os
import importlib
import fairseq
from fairseq.tasks import FairseqTask
from hydra.core.config_store import ConfigStore # dynamic configuration management


TASK_REGISTRY = {}
TASK_CLASS_NAMES = set()
TASK_DATACLASS_REGISTRY = {}

def register_task(name, dataclass=None):
    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError('Cannot register duplicate task ({name})')
        if not issubclass(cls, FairseqTask):
            raise ValueError('Task ({name}: {cls.__name__}) must extend FairseqTask')
        if cls.__name__ in TASK_CLASS_NAMES:
            raise ValueError('Cannot register task with duplicate class name ({cls.__name__})')
        TASK_REGISTRY[name] = cls
        TASK_CLASS_NAMES.add(cls.__name__)

        if dataclass is not None and not issubclass(dataclass, FairseqDataclass):
            raise ValueError(
                "Dataclass {} must extend FairseqDataclass".format(dataclass)
            )

        cls.__dataclass = dataclass
        if dataclass is not None:
            TASK_DATACLASS_REGISTRY[name] = dataclass

            cs = ConfigStore.instance() # dynamic management of configuration
            node = dataclass()
            node._name = name
            cs.store(name=name, group="task", node=node, provider="fairseq")

        return cls

    return register_task_cls