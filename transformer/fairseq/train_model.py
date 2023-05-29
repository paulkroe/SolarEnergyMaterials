import  argparse
import fairseq
from task import register_task
from fairseq.tasks import FairseqTask
from transformer.dataset import ogbGraphormerDataset
import sys


# sys.path.append('../../transformer/fairseq/task')


# setsup task (e.g., load dictionaries)

# dataset_spec either "ogbg-molhiv", "ogbg-molpcba", "pcqm4mv2", "pcqm4m"

@register_task('GraphPrediction')
class GraphPredictionTask(fairseq.tasks.FairseqTask):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        if getattr(self.config, self.config.dataset_name, None) is None or getattr(self.config, self.config.seed, None) is None:
            raise ValueError("Dataset name, seed not specified")

        self.dm = ogbGraphormerDataset(
            dataset_spec=config.datasset_name,
            seed=config.seed
        )

if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument("test_model", help="one forward pass through transformer model", type=str)
    parser.add_argument("--src_vocab", help="size of source vocab", type=int)
    parser.add_argument("--tgt_vocab", help="size of target vocab", type=int)
    parser.add_argument("--N", help="Number of encoder layers", type=int, default=4)
    parser.add_argument("--d_model", help="Dimension of model", type=int, default=32)
    parser.add_argument("--d_ff", help="Dimension of feed forward layer", type=int, default=64)
    parser.add_argument("--h", help="Number of heads", type=int, default=4)
    parser.add_argument("--dropout", help="Dropout probability", type=float, default=0.1)
    args = parser.parse_args()
    if args.test_model == "test_model":
        if args.src_vocab is None:
            args.src_vocab = int(input("Enter source vocab size: "))
        if args.tgt_vocab is None:
            args.tgt_vocab = int(input("Enter target vocab size: "))
        
        # setup task
        task = fairseq.tasks.setup_task(args)

        # build model and criterion
        model = task.build_model(args)
        criterion = task.build_criterion(args)

        # load datasets
        task.load_dataset('train')
        task.load_dataset('valid')

        # iterate over mini-batches of data
        batch_itr = task.get_batch_iterator(
            task.dataset('train'), max_tokens=4096,
        )
        for batch in batch_itr:
            # compute the loss
            loss, sample_size, logging_output = task.get_loss(
                model, criterion, batch,
            )
            loss.backward()