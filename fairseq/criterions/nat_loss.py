# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from torch import Tensor

from dataclasses import dataclass, field


@dataclass
class LabelSmoothedDualImitationCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )


@register_criterion("nat_loss", dataclass=LabelSmoothedDualImitationCriterionConfig)
class LabelSmoothedDualImitationCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing):
        super().__init__(task)
        self.label_smoothing = label_smoothing

    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0, weight=None,
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():  # all False
            # nll_loss = torch.tensor(0)
            nll_loss = outputs.new_tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                # print("weight", weight)
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none", weight=weight)
            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")  # pointwise
                # losses[:, :, 0] = losses[:, :, 0] * rescale0  # rescale class 0
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                    nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    ## not very smart way to print accuracy... but works
    def _print_accuracy(self, names, outputs, obj):
        dels, plds = None, None
        if "del" in names and outputs[obj].get("aggr_del", False):
            # check del accuracy
            del_masks = outputs[obj].get("mask")
            del_outputs = outputs[obj].get("out")

            # print('mean of del pred after mask', torch.mean(del_outputs[del_masks]))
            # print('std:', torch.std(del_outputs[del_masks]))

            ## apply delete threshold during validation
            # delete_threshold = 0.2
            # if delete_threshold != 0.0:
            #     del_outputs = F.softmax(del_outputs, -1)
            #     del_outputs = ((del_outputs[:,:, 1] - del_outputs[:,:, 0]) > delete_threshold).squeeze(-1)
            #     # print('avg score: ', torch.mean(word_del_score, dim=(0,1)).tolist())
            #     # print('after threshold: ', word_del_pred.sum().item())
            #     # print('original: ', word_del_score.max(-1)[1].bool().sum().item())
            # else:
            #     del_outputs = del_outputs.max(-1)[1].bool()
            
            del_outputs = del_outputs.max(-1)[1].bool()

            del_targets = outputs[obj].get("tgt")

            # print('del pred\n{}'.format(del_outputs))
            # print('del mask\n{}'.format(del_masks))
            # print('del target\n{}'.format(del_targets))
            
            del_outputs, del_targets = del_outputs[del_masks], del_targets[del_masks]
            del_acc = torch.sum(del_outputs == del_targets) / torch.sum(del_masks)
            del_recall = torch.sum(torch.logical_and(del_targets == 1, del_outputs == del_targets)) / torch.sum(del_targets)
            del_prec = torch.sum(torch.logical_and(del_targets == 1, del_outputs == del_targets)) / torch.sum(del_outputs)
            # print('masked del pred\n{}'.format(del_outputs))
            # print('masked del target\n{}'.format(del_targets))
            #print('==?: ', del_outputs == del_targets)

            #print('batch: ', del_masks.size(0))
            print('del_acc: ', del_acc.item())
            print('del_recall: ', del_recall.item())
            print('del_prec: ', del_prec.item())
            print('del_num: ', torch.sum(del_masks).item() / del_masks.size(0))

            dels = {
                'del_acc': del_acc.item(),
                'del_recall': del_recall.item(),
                'del_prec': del_prec.item()
            }
        
        if "pld" in names and outputs[obj].get("ls", False) and not outputs[obj].get("nll_loss", False):
            # check pld accuracy
            pld_masks = outputs[obj].get("mask")
            pld_outputs = outputs[obj].get("out")

            pld_outputs = pld_outputs.max(-1)[1]
            pld_targets = outputs[obj].get("tgt")

            pld_outputs, pld_targets = pld_outputs[pld_masks], pld_targets[pld_masks]
            pld_acc = torch.sum(pld_outputs == pld_targets) / torch.sum(pld_masks)
            pld_l1 = torch.sum(torch.abs(pld_outputs - pld_targets)) / torch.sum(pld_masks)
            pld_l1_positif = torch.sum(torch.max(pld_outputs - pld_targets, torch.tensor(0))) / torch.sum(pld_masks)
            pld_l1_negatif = torch.sum(torch.min(pld_outputs - pld_targets, torch.tensor(0))) / torch.sum(pld_masks)
            pld_l1_sent = torch.sum(torch.abs(pld_outputs - pld_targets)) / pld_masks.size(0)
            
            print('batch: ', pld_masks.size(0))
            print('pld_acc: ', pld_acc.item())
            print('pld_l1: ', pld_l1.item())
            print('pld_l1_positif: ', pld_l1_positif.item())
            print('pld_l1_negatif: ', pld_l1_negatif.item())
            print('pld_l1_sent: ', pld_l1_sent.item())
            print('pld_num: ', torch.sum(pld_masks).item() / pld_masks.size(0))

            plds = {
                "pld_acc": pld_acc.item(),
                "pld_l1": pld_l1.item()
            }

        if "tok" in names and outputs[obj].get("nll_loss", False):
            # check tok accuracy
            tok_masks = outputs[obj].get("mask")
            tok_outputs = outputs[obj].get("out")
            tok_targets = outputs[obj].get("tgt")

            tok_outputs = tok_outputs.max(-1)[1]
            tok_outputs, tok_targets = tok_outputs[tok_masks], tok_targets[tok_masks]
            tok_acc = torch.sum(tok_outputs == tok_targets) / torch.sum(tok_masks)

            print('tok_acc: ', tok_acc.item())
            print('tok_num: ', torch.sum(tok_masks).item() / tok_masks.size(0))

        return dels, plds

    def forward(self, model, sample, reduce=True, 
        aggravate=False, sampled_step=None, word_predictions=None, new_del_input=False, save_tensors=False,
    ):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        # print("PREV TARGET DEVICE: {}\tTARGET DEVICE: {}\tEXAMPLE IDS: {}".format(prev_output_tokens.device, tgt_tokens.device, sample["id"]))

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens, 
                    aggravate=aggravate, sampled_step=sampled_step, word_predictions=word_predictions, new_del_input=new_del_input,
                )

        losses, nll_loss = [], []

        for obj in outputs:
            if outputs[obj].get("loss", None) is None:  # all 3
                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                    weight=outputs[obj].get("weight", None),
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )

            losses += [_losses]

            if outputs[obj].get("nll_loss", False):  # tok
                nll_loss += [_losses.get("nll_loss", 0.0)]

            # # print accuracy 
            # self._print_accuracy(["del", "pld", "tok"], outputs, obj)

            if save_tensors and outputs[obj].get("sample3_mask", False):
                # get pld labels and preds
                pld_masks = outputs[obj].get("mask")
                pld_labels = outputs[obj].get("tgt")
                pld_outputs = outputs[obj].get("out")

                pld_preds = pld_outputs.max(-1)[1]
                pld_proba = F.softmax(pld_outputs, -1)
                pld_esperance = torch.matmul(pld_proba, torch.arange(pld_outputs.size(-1), device='cuda', dtype=pld_proba.dtype))
                top5_proba, top5_len = torch.topk(pld_proba, 5, dim=-1, largest=True, sorted=True)

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
          
        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        # print("DEVICE: {}\tLOGGING OUTPUT: {}".format(tgt_tokens.device, logging_output))
        if save_tensors:
            return loss, sample_size, logging_output, (pld_masks, pld_labels, pld_preds, pld_esperance, top5_proba, top5_len)
        else:
            return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
