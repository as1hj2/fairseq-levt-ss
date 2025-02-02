# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import torch
from typing import Optional
from fairseq import utils
from fairseq.data import LanguagePairDataset
from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import register_task
from fairseq.tasks.translation import (
    TranslationConfig,
    TranslationTask,
    load_langpair_dataset,
)
from fairseq.utils import new_arange


NOISE_CHOICES = ChoiceEnum(["random_delete", "random_mask", "no_noise", "full_mask"])


@dataclass
class TranslationLevenshteinConfig(TranslationConfig):
    noise: NOISE_CHOICES = field(
        default="random_delete",
        metadata={"help": "type of noise"},
    )
    prev_target: Optional[str] = field(
        default=None, metadata={"help": "prev target language"}
    )
    prev_target_prob: float = field(
        default=0.0,
        metadata={"help": "probability to use prev target instead of generate on the fly"}
    )
    use_aggravate_prob: float = field(
        default=0.5,
        metadata={"help": "probability of using aggravate"}
    )
    sample1_prob: float = field(
        default=0.5,
        metadata={"help": "probability of sample first iteration (pld, tok)"}
    )
    new_del_input: bool = field(
        default=False,
        metadata={"help": "if True, give [empty -> exp pld -> tok pred] to del pred"}
    )


@register_task("translation_lev", dataclass=TranslationLevenshteinConfig)
class TranslationLevenshteinTask(TranslationTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    cfg: TranslationLevenshteinConfig

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            prepend_bos=True,
            prev_target=self.cfg.prev_target,
        )

    def inject_noise(self, target_tokens):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
            )
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True
            )

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = (
                2
                + (
                    (target_length - 2)
                    * target_score.new_zeros(target_score.size(0), 1).uniform_()
                ).long()
            )
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = (
                target_tokens.gather(1, target_rank)
                .masked_fill_(target_cutoff, pad)
                .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
            )
            prev_target_tokens = prev_target_tokens[
                :, : prev_target_tokens.ne(pad).sum(1).max()
            ]

            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk
            )
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = (
                target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
            )
            return target_tokens.masked_fill(~target_mask, unk)

        if self.cfg.noise == "random_delete":
            return _random_delete(target_tokens)
        elif self.cfg.noise == "random_mask":
            return _random_mask(target_tokens)
        elif self.cfg.noise == "full_mask":
            return _full_mask(target_tokens)
        elif self.cfg.noise == "no_noise":
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, models, args, **unused):
        # add models input to match the API for SequenceGenerator
        from fairseq.iterative_refinement_generator import IterativeRefinementGenerator

        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
            max_iter=getattr(args, "iter_decode_max_iter", 10),
            beam_size=getattr(args, "iter_decode_with_beam", 1),
            reranking=getattr(args, "iter_decode_with_external_reranker", False),
            decoding_format=getattr(args, "decoding_format", None),
            adaptive=not getattr(args, "iter_decode_force_max_iter", False),
            retain_history=getattr(args, "retain_iter_history", False),
            delete_threshold=getattr(args, "delete_threshold", 0.0),
            use_pld_dp=getattr(args, "use_pld_dp", False),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None, prefix=None, initial_lens=None, initial_tokens=None,):
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError(
                "Constrained decoding with the translation_lev task is not supported"
            )

        return LanguagePairDataset(
            src_tokens, src_lengths, self.source_dictionary, append_bos=True, prefix=prefix, initial_lens=initial_lens, initial_tokens=initial_tokens,
        )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        aggravate = torch.rand(1) >= self.cfg.use_aggravate_prob
        # aggravate = True
        # print('aggravate: ', aggravate)

        if not aggravate:
            sample["prev_target"] = self.inject_noise(sample["target"])
            # sample["prev_target"] = torch.tensor([[0,6,6,6,6,6,6,2]], dtype=torch.int)
            sampled_step = None
            word_predictions = None

        else:
            # assert sample.get("prev_target", None) is not None
            # initialize <bos><eos>
            sample["prev_target"] = sample["target"].new_zeros(sample["target"].size(0), 2)
            sample["prev_target"][:, 0] = self.tgt_dict.bos()
            sample["prev_target"][:, 1] = self.tgt_dict.eos()

            # print("sample[prev_target]:\n{}".format(sample["prev_target"]))
            # print("sample[target]:\n{}".format(sample["target"]))

            sample1 = torch.rand(1) >= self.cfg.sample1_prob
            sampled_step = 1 if sample1 else 3
            # sampled_step = 3
            
            # print('update_num: ', update_num)
            # print('sampled_step: ', sampled_step)

            # if sampled_step == 1 and self.cfg.new_del_input:
            if self.cfg.new_del_input:
                word_predictions = None
            else:
                model.eval()
                word_predictions = model.pre_predict(sample["net_input"]["src_tokens"], sample["net_input"]["src_lengths"], sample["prev_target"], sample["target"], sampled_step)
                
            # print("Using prev target data")
            # sample["prev_target"] = sample["target"]
            # if update_num > 270:
            #     print("STEP {} sample id: {}".format(update_num, sample["id"]))

        model.train()
        loss, sample_size, logging_output = criterion(model, sample, aggravate=aggravate, sampled_step=sampled_step, word_predictions=word_predictions, new_del_input=self.cfg.new_del_input)
        
        # print('logging_output: ', logging_output)

        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion, save_tensors=False):
        model.eval()
        with torch.no_grad():
            # if torch.rand(1) >= self.cfg.prev_target_prob:
            #     sample["prev_target"] = self.inject_noise(sample["target"])
            # else:
            #     assert sample.get("prev_target", None) is not None

            # aggravate
            sample["prev_target"] = sample["target"].new_zeros(sample["target"].size(0), 2)
            sample["prev_target"][:, 0] = self.tgt_dict.bos()
            sample["prev_target"][:, 1] = self.tgt_dict.eos()

            # defaut setting, should manually change here
            sampled_step = 3
            new_del_input = True
            
            # torch.set_printoptions(profile="full")
            # print('prev_predictions:\n{}'.format(word_predictions))
            # print('target:\n{}'.format(sample["target"]))
            if new_del_input:
                word_predictions = None
            else:
                word_predictions = model.pre_predict(sample["net_input"]["src_tokens"], sample["net_input"]["src_lengths"], sample["prev_target"], sample["target"], sampled_step)

            if save_tensors:
                loss, sample_size, logging_output, pld_tensors = criterion(model, sample, aggravate=True, sampled_step=sampled_step, word_predictions=word_predictions, new_del_input=new_del_input, save_tensors=save_tensors)
                return loss, sample_size, logging_output, pld_tensors
            else:
                loss, sample_size, logging_output = criterion(model, sample, aggravate=True, sampled_step=sampled_step, word_predictions=word_predictions, new_del_input=True, save_tensors=save_tensors)
                return loss, sample_size, logging_output
