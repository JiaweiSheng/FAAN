from collections import defaultdict
from torch import optim
from collections import deque
from args import read_options
from data_loader import *
from matcher import *
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm


class Trainer(object):
    def __init__(self, arg):
        super(Trainer, self).__init__()
        for k, v in vars(arg).items():
            setattr(self, k, v)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.meta = not self.no_meta

        # pre-train
        if self.random_embed:
            use_pretrain = False
        else:
            use_pretrain = True

        logging.info('LOADING SYMBOL ID AND SYMBOL EMBEDDING')
        if self.test or self.random_embed:
            # gen symbol2id, without embedding
            self.load_symbol2id()
            use_pretrain = False
        else:
            self.load_embed()
        self.use_pretrain = use_pretrain

        self.num_symbols = len(self.symbol2id.keys()) - 1  # one for 'PAD'
        self.pad_id = self.num_symbols

        self.Matcher = Matcher(self.embed_dim, self.num_symbols,
                               use_pretrain=self.use_pretrain,
                               embed=self.symbol2vec,
                               dropout_layers=self.dropout_layers,
                               dropout_input=self.dropout_input,
                               dropout_neighbors=self.dropout_neighbors,
                               finetune=self.fine_tune,
                               num_transformer_layers=self.num_transformer_layers,
                               num_transformer_heads=self.num_transformer_heads,
                               device=self.device
                               )

        self.Matcher.to(self.device)
        self.batch_nums = 0
        if self.test:
            self.writer = None
        else:
            self.writer = SummaryWriter('logs/' + self.prefix)

        self.parameters = filter(lambda p: p.requires_grad, self.Matcher.parameters())

        self.optim = optim.Adam(self.parameters, lr=self.lr, weight_decay=self.weight_decay)
        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        self.num_ents = len(self.ent2id.keys())

        logging.info('BUILDING CONNECTION MATRIX')
        degrees = self.build_connection(max_=self.max_neighbor)

        logging.info('LOADING CANDIDATES ENTITIES')
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json'))

        # load answer dict
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))

    def load_symbol2id(self):
        # gen symbol2id, without embedding
        symbol_id = {}
        rel2id = json.load(open(self.dataset + '/relation2ids'))
        ent2id = json.load(open(self.dataset + '/ent2ids'))
        i = 0
        # rel and ent combine together
        for key in rel2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        symbol_id['PAD'] = i
        self.symbol2id = symbol_id
        self.symbol2vec = None

    def load_embed(self):
        # gen symbol2id, with embedding
        symbol_id = {}
        rel2id = json.load(open(self.dataset + '/relation2ids'))  # relation2id contains inverse rel
        ent2id = json.load(open(self.dataset + '/ent2ids'))

        logging.info('LOADING PRE-TRAINED EMBEDDING')
        if self.embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL']:
            ent_embed = np.loadtxt(self.dataset + '/entity2vec.' + self.embed_model)
            rel_embed = np.loadtxt(self.dataset + '/relation2vec.' + self.embed_model)  # contain inverse edge

            if self.embed_model == 'ComplEx':
                # normalize the complex embeddings
                ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
                ent_std = np.std(ent_embed, axis=1, keepdims=True)
                rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
                rel_std = np.std(rel_embed, axis=1, keepdims=True)
                eps = 1e-3
                ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
                rel_embed = (rel_embed - rel_mean) / (rel_std + eps)

            assert ent_embed.shape[0] == len(ent2id.keys())
            assert rel_embed.shape[0] == len(rel2id.keys())

            i = 0
            embeddings = []
            for key in rel2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(rel_embed[rel2id[key], :]))

            for key in ent2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(ent_embed[ent2id[key], :]))

            symbol_id['PAD'] = i
            embeddings.append(list(np.zeros((rel_embed.shape[1],))))
            embeddings = np.array(embeddings)
            assert embeddings.shape[0] == len(symbol_id.keys())

            self.symbol2id = symbol_id
            self.symbol2vec = embeddings

    def build_connection(self, max_=100):
        self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)
        with open(self.dataset + '/path_graph') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1, rel, e2 = line.rstrip().split()
                self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2]))  # 1-n
                self.e1_rele2[e2].append((self.symbol2id[rel + '_inv'], self.symbol2id[e1]))  # n-1

        degrees = {}
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            degrees[ent] = len(neighbors)
            self.e1_degrees[id_] = len(neighbors)  # add one for self conn
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]  # rel
                self.connections[id_, idx, 1] = _[1]  # tail
        return degrees

    def save(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.Matcher.state_dict(), path)

    def load(self, path=None):
        if path:
            self.Matcher.load_state_dict(torch.load(path))
        else:
            self.Matcher.load_state_dict(torch.load(self.save_path))

    def get_meta(self, left, right):
        left_connections = Variable(
            torch.LongTensor(np.stack([self.connections[_, :, :] for _ in left], axis=0))).to(self.device)
        left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).to(self.device)
        right_connections = Variable(
            torch.LongTensor(np.stack([self.connections[_, :, :] for _ in right], axis=0))).to(self.device)
        right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).to(self.device)
        return (left_connections, left_degrees, right_connections, right_degrees)

    def train(self):
        logging.info('START TRAINING...')
        best_mrr = 0.0
        best_batches = 0

        losses = deque([], self.log_every)
        margins = deque([], self.log_every)
        for data in train_generate(self.dataset, self.batch_size, self.train_few, self.symbol2id, self.ent2id,
                                   self.e1rel_e2):
            support, query, false, support_left, support_right, query_left, query_right, false_left, false_right = data

            self.batch_nums += 1
            support_meta = self.get_meta(support_left, support_right)
            query_meta = self.get_meta(query_left, query_right)
            false_meta = self.get_meta(false_left, false_right)

            support = Variable(torch.LongTensor(support)).to(self.device)
            query = Variable(torch.LongTensor(query)).to(self.device)
            false = Variable(torch.LongTensor(false)).to(self.device)
            self.Matcher.train()
            if self.no_meta:
                positive_score, negative_score = self.Matcher(support, query, false, isEval=False)
            else:
                positive_score, negative_score = self.Matcher(support, query, false, isEval=False,
                                                              support_meta=support_meta,
                                                              query_meta=query_meta,
                                                              false_meta=false_meta)
            margin_ = positive_score - negative_score
            loss = F.relu(self.margin - margin_).mean()
            margins.append(margin_.mean().item())
            lr = adjust_learning_rate(optimizer=self.optim, epoch=self.batch_nums, lr=self.lr,
                                      warm_up_step=self.warm_up_step,
                                      max_update_step=self.max_batches)
            losses.append(loss.item())

            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(self.parameters, self.grad_clip)
            self.optim.step()
            if self.batch_nums % self.log_every == 0:
                lr = self.optim.param_groups[0]['lr']
                logging.info(
                    'Batch: {:d}, Avg_batch_loss: {:.6f}, lr: {:.6f}, '.format(
                        self.batch_nums,
                        np.mean(losses),
                        lr))
                self.writer.add_scalar('Avg_batch_loss_every_log', np.mean(losses), self.batch_nums)

            if self.batch_nums % self.eval_every == 0:
                logging.info('Batch_nums is %d' % self.batch_nums)
                hits10, hits5, hits1, mrr = self.eval(meta=self.meta, mode='dev')
                self.writer.add_scalar('HITS10', hits10, self.batch_nums)
                self.writer.add_scalar('HITS5', hits5, self.batch_nums)
                self.writer.add_scalar('HITS1', hits1, self.batch_nums)
                self.writer.add_scalar('MRR', mrr, self.batch_nums)
                self.save()

                if mrr > best_mrr:
                    self.save(self.save_path + '_best')
                    best_mrr = mrr
                    best_batches = self.batch_nums
                logging.info('Best_mrr is {:.6f}, when batch num is {:d}'.format(best_mrr, best_batches))

            if self.batch_nums == self.max_batches:
                self.save()
                break

            if self.batch_nums - best_batches > self.eval_every * 10:
                logging.info('Early stop!')
                self.save()
                break

    def eval(self, mode='dev', meta=False):
        self.Matcher.eval()

        symbol2id = self.symbol2id
        few = self.few

        logging.info('EVALUATING ON %s DATA' % mode.upper())
        if mode == 'dev':
            test_tasks = json.load(open(self.dataset + '/dev_tasks.json'))
        else:
            test_tasks = json.load(open(self.dataset + '/test_tasks.json'))

        rel2candidates = self.rel2candidates

        hits10 = []
        hits5 = []
        hits1 = []
        mrr = []
        for query_ in test_tasks.keys():
            hits10_ = []
            hits5_ = []
            hits1_ = []
            mrr_ = []
            candidates = rel2candidates[query_]
            support_triples = test_tasks[query_][:few]
            support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]

            if meta:
                support_left = [self.ent2id[triple[0]] for triple in support_triples]
                support_right = [self.ent2id[triple[2]] for triple in support_triples]
                support_meta = self.get_meta(support_left, support_right)

            support = Variable(torch.LongTensor(support_pairs)).to(self.device)

            for triple in test_tasks[query_][few:]:
                true = triple[2]
                query_pairs = []
                query_pairs.append([symbol2id[triple[0]], symbol2id[triple[2]]])
                if meta:
                    query_left = []
                    query_right = []
                    query_left.append(self.ent2id[triple[0]])
                    query_right.append(self.ent2id[triple[2]])
                for ent in candidates:
                    if (ent not in self.e1rel_e2[triple[0] + triple[1]]) and ent != true:
                        query_pairs.append([symbol2id[triple[0]], symbol2id[ent]])
                        if meta:
                            query_left.append(self.ent2id[triple[0]])
                            query_right.append(self.ent2id[ent])

                query = Variable(torch.LongTensor(query_pairs)).to(self.device)

                if meta:
                    query_meta = self.get_meta(query_left, query_right)
                    scores, _ = self.Matcher(support, query, None, isEval=True,
                                             support_meta=support_meta,
                                             query_meta=query_meta,
                                             false_meta=None)
                    scores.detach()
                    scores = scores.data

                scores = scores.cpu().numpy()
                sort = list(np.argsort(scores, kind='stable'))[::-1]
                rank = sort.index(0) + 1
                if rank <= 10:
                    hits10.append(1.0)
                    hits10_.append(1.0)
                else:
                    hits10.append(0.0)
                    hits10_.append(0.0)
                if rank <= 5:
                    hits5.append(1.0)
                    hits5_.append(1.0)
                else:
                    hits5.append(0.0)
                    hits5_.append(0.0)
                if rank <= 1:
                    hits1.append(1.0)
                    hits1_.append(1.0)
                else:
                    hits1.append(0.0)
                    hits1_.append(0.0)
                mrr.append(1.0 / rank)
                mrr_.append(1.0 / rank)

            logging.critical('{} Hits10:{:.3f}, Hits5:{:.3f}, Hits1:{:.3f}, MRR:{:.3f}'.format(query_,
                                                                                               np.mean(
                                                                                                   hits10_),
                                                                                               np.mean(hits5_),
                                                                                               np.mean(hits1_),
                                                                                               np.mean(mrr_),
                                                                                               ))
            logging.info('Number of candidates: {}, number of test examples {}'.format(len(candidates), len(hits10_)))
        logging.critical('HITS10: {:.3f}'.format(np.mean(hits10)))
        logging.critical('HITS5: {:.3f}'.format(np.mean(hits5)))
        logging.critical('HITS1: {:.3f}'.format(np.mean(hits1)))
        logging.critical('MRR: {:.3f}'.format(np.mean(mrr)))
        return np.mean(hits10), np.mean(hits5), np.mean(hits1), np.mean(mrr)

    def test_(self, path=None):
        self.load(path)
        logging.info('Pre-trained model loaded for test')
        self.eval(mode='test', meta=self.meta)

    def eval_(self, path=None):
        self.load(path)
        logging.info('Pre-trained model loaded for dev')
        self.eval(mode='dev', meta=self.meta)


def seed_everything(seed=2040):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def adjust_learning_rate(optimizer, epoch, lr, warm_up_step, max_update_step, end_learning_rate=0.0, power=1.0):
    epoch += 1
    if warm_up_step > 0 and epoch <= warm_up_step:
        warm_up_factor = epoch / float(warm_up_step)
        lr = warm_up_factor * lr
    elif epoch >= max_update_step:
        lr = end_learning_rate
    else:
        lr_range = lr - end_learning_rate
        pct_remaining = 1 - (epoch - warm_up_step) / (max_update_step - warm_up_step)
        lr = lr_range * (pct_remaining ** power) + end_learning_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    args = read_options()
    if not os.path.exists('./logs_'):
        os.mkdir('./logs_')
    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler('./logs_/log-{}.txt'.format(args.prefix))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    seed_everything(args.seed)

    logging.info('*' * 100)
    logging.info('*** hyper-parameters ***')
    for k, v in vars(args).items():
        logging.info(k + ': ' + str(v))
    logging.info('*' * 100)

    trainer = Trainer(args)

    if args.test:
        trainer.test_()
        trainer.eval_()
    else:
        trainer.train()
        print('last checkpoint!')
        trainer.eval_()
        trainer.test_()
        print('best checkpoint!')
        trainer.eval_(args.save_path + '_best')
        trainer.test_(args.save_path + '_best')
