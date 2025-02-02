# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from base64 import encode
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATDecoder, FairseqNATModel, ensemble_decoder
from fairseq.models.transformer import Embedding
from fairseq.modules import TransformerDecoderLayer
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .levenshtein_utils import (
    _apply_del_words,
    _apply_ins_masks,
    _apply_ins_words,
    _fill,
    _get_del_targets,
    _get_ins_targets,
    _skip,
    _skip_encoder_out,
    _get_pld_len,
)


@register_model("levenshtein_transformer")
class LevenshteinTransformerModel(FairseqNATModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.prev_del = args.prev_del
        self.post_del = args.post_del

    @property
    def allow_length_beam(self):
        return False

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)
        parser.add_argument(
            "--early-exit",
            default="6,6,6",
            type=str,
            help="number of decoder layers before word_del, mask_ins, word_ins",
        )
        parser.add_argument(
            "--no-share-discriminator",
            action="store_true",
            help="separate parameters for discriminator",
        )
        parser.add_argument(
            "--no-share-maskpredictor",
            action="store_true",
            help="separate parameters for mask-predictor",
        )
        parser.add_argument(
            "--share-discriminator-maskpredictor",
            action="store_true",
            help="share the parameters for both mask-predictor and discriminator",
        )
        parser.add_argument(
            "--sampling-for-deletion",
            action="store_true",
            help="instead of argmax, use sampling to predict the tokens",
        )
        parser.add_argument(
            "--prev-del",
            action="store_true",
            help="apply deletion before insertion"
        )
        parser.add_argument(
            "--prediction-for-prev-deletion",
            action="store_true",
            help="use predicted and reference to delete tokens in prev deletion",
        )
        parser.add_argument(
            "--post-del",
            action="store_true",
            help="apply deletion after insertion"
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = LevenshteinTransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, 
        aggravate=False, sampled_step=None, word_predictions=None, new_del_input=False,
        **kwargs
    ):

        assert tgt_tokens is not None, "forward function only supports training."

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # print("DEVICE: {}\ttgt_tokens:\n{}".format(tgt_tokens.device, self.tgt_dict.string(tgt_tokens)))
        # print("DEVICE: {}\tprev_outpout_tokens:\n{}".format(prev_output_tokens.device, self.tgt_dict.string(prev_output_tokens)))

        # applying deletion before insertion
        if self.prev_del:
            # generate training labels for deletion
            prev_word_del_targets = _get_del_targets(prev_output_tokens, tgt_tokens, self.pad)
            prev_word_del_out, _ = self.decoder.forward_word_del(
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out,
            )
            prev_word_del_masks = prev_output_tokens.ne(self.pad)

            # delete tokens
            prev_output_tokens, _, _ = _apply_del_words(
                prev_output_tokens,
                in_scores=None,
                in_attn=None,
                word_del_pred=prev_word_del_targets.bool() | 
                prev_word_del_out.max(-1)[1].bool() if self.predict_prev_del # delete both pred and exp
                else prev_word_del_targets.bool(),
                padding_idx=self.pad,
                bos_idx=self.bos,
                eos_idx=self.eos,
            )

            # delete unnecessary paddings
            cut_off = prev_output_tokens.ne(self.pad).sum(1).max()
            prev_output_tokens = prev_output_tokens[:, :cut_off]

            # print("prediction_prev_del={}".format(self.predict_prev_del))
            # print("prev_word_del_targets:\n{}".format(prev_word_del_targets.bool()))
            # # print("prev_word_del_out:\n{}".format(prev_word_del_out))
            # print("prev_word_del_out_pred:\n{}".format(prev_word_del_out.max(-1)[1].bool()))
            # print("prev_word_del_targets | prev_word_del_out_pred:\n{}".format(prev_word_del_targets.bool() | prev_word_del_out.max(-1)[1].bool()))
            # print("prev_word_del_masks:\n{}".format(prev_word_del_masks))
            # print("prev_output_tokens after deletion:\n{}".format(self.tgt_dict.string(prev_output_tokens)))

        if not aggravate:
            # generate training labels for insertion
            # masked_tgt_masks=mask to compute loss only on pld positions; masked_tgt_tokens=applied pld ins; mask_ins_targets=pld num pred
            masked_tgt_masks, masked_tgt_tokens, mask_ins_targets, inserted_tgt_tokens = _get_ins_targets(
                prev_output_tokens, tgt_tokens, self.pad, self.unk, aggravate=False
            )
            mask_ins_targets = mask_ins_targets.clamp(min=0, max=255)  # for safe prediction
            mask_ins_masks = prev_output_tokens[:, 1:].ne(self.pad)
            
            # torch.set_printoptions(profile="full")
            # print("masked_tgt_masks:\n{}".format(masked_tgt_masks))
            # print("masked_tgt_tokens:\n{}".format(self.tgt_dict.string(masked_tgt_tokens)))
            # print("masked_tgt_tokens:\n{}".format(masked_tgt_tokens))
            # print("mask_ins_targets:\n{}".format(mask_ins_targets))
            # print("mask_ins_masks:\n{}".format(mask_ins_masks))

            # print('before pld pred: ', torch.cuda.memory_allocated())
            mask_ins_out, _ = self.decoder.forward_mask_ins(
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out,
            )
            # print("mask_ins_out:\n{}".format(mask_ins_out))
            # print('after pld pred: ', torch.cuda.memory_allocated())

            word_ins_out, _ = self.decoder.forward_word_ins( # pred for all positions
                normalize=False,
                prev_output_tokens=masked_tgt_tokens,
                encoder_out=encoder_out,
            )
            # print("word_ins_out:\n{}".format(word_ins_out))
            # print('after tok pred: ', torch.cuda.memory_allocated())

            # make online prediction
            if self.decoder.sampling_for_deletion:
                word_predictions = torch.multinomial(
                    F.softmax(word_ins_out, -1).view(-1, word_ins_out.size(-1)), 1
                ).view(word_ins_out.size(0), -1)
            else:
                word_predictions = F.log_softmax(word_ins_out, dim=-1).max(2)[1]

            # print("word_predictions:\n{}".format(word_predictions))
            word_predictions.masked_scatter_( # replace unecessary positions by previous tokens
                ~masked_tgt_masks, tgt_tokens[~masked_tgt_masks]
            )
            # print("word_predictions after masked_scatter_:\n{}".format(word_predictions))

            if self.post_del:
                # generate training labels for deletion
                word_del_targets = _get_del_targets(word_predictions, tgt_tokens, self.pad)
                word_del_out, _ = self.decoder.forward_word_del(
                    normalize=False,
                    prev_output_tokens=word_predictions,
                    encoder_out=encoder_out,
                )
                word_del_masks = word_predictions.ne(self.pad)
                # print("word_del_targets:\n{}".format(word_del_targets))
                # print("word_del_out:\n{}".format(word_del_out.argmax(-1)))
                # print("word_del_masks:\n{}".format(word_del_masks))
                # print('after del pred: ', torch.cuda.memory_allocated())

                output = {
                "mask_ins": {
                    "out": mask_ins_out,
                    "tgt": mask_ins_targets,
                    "mask": mask_ins_masks,
                    "ls": 0.01,
                },
                "word_ins": {
                    "out": word_ins_out,
                    "tgt": tgt_tokens,
                    "mask": masked_tgt_masks,
                    "ls": self.args.label_smoothing,
                    "nll_loss": True,
                },
            }
            if self.post_del:
                output["word_del"] = {
                    "out": word_del_out,
                    "tgt": word_del_targets,
                    "mask": word_del_masks,
                    "aggr_del": True,
                }
            if self.prev_del:
                output["prev_word_del"] = {
                    "out": prev_word_del_out,
                    "tgt": prev_word_del_targets,
                    "mask": prev_word_del_masks,
                }

            return output

        # if use AggravaTe
        else:
            output = {}

            # print('before pld pred: ', torch.cuda.memory_allocated())
            if sampled_step == 1 or new_del_input:
                # [pld] pred
                mask_ins_out, _ = self.decoder.forward_mask_ins(
                    normalize=False,
                    prev_output_tokens=prev_output_tokens,
                    encoder_out=encoder_out,
                )
                # print("mask_ins_out:\n{}".format(mask_ins_out), mask_ins_out.shape)
                # print('after pld pred: ', torch.cuda.memory_allocated())
        
                # [pld] label for training, apply exp [pld]
                ## masked_tgt_masks=mask to compute loss only on pld positions; in+out
                ## masked_tgt_tokens=applied pld ins; in+out
                ## mask_ins_targets=pld num pred; in-1
                ## inserted_tgt_tokens; in+out
                masked_tgt_masks, masked_tgt_tokens, mask_ins_targets, inserted_tgt_tokens = _get_ins_targets(
                    prev_output_tokens, tgt_tokens, self.pad, self.unk, aggravate=True
                )
                # print("before:masked_tgt_masks:\n{}".format(masked_tgt_masks))
                # print("before:masked_tgt_tokens:\n{}".format(self.tgt_dict.string(masked_tgt_tokens)), masked_tgt_tokens)
                # print('before:inserted_tgt_tokens', inserted_tgt_tokens)

                # cutoff unnecessary paddings
                cut_off = masked_tgt_tokens.ne(self.pad).sum(1).max()
                masked_tgt_tokens = masked_tgt_tokens[:, :cut_off]
                masked_tgt_masks = masked_tgt_masks[:, :cut_off]
                inserted_tgt_tokens = inserted_tgt_tokens[:, :cut_off]

                mask_ins_targets = mask_ins_targets.clamp(min=0, max=255)  # for safe prediction
                mask_ins_masks = prev_output_tokens[:, 1:].ne(self.pad)  # input without eos and pad, true num of ins labels
                # print("masked_tgt_masks:\n{}".format(masked_tgt_masks))
                # print("masked_tgt_tokens:\n{}".format(self.tgt_dict.string(masked_tgt_tokens)), masked_tgt_tokens)
                # print("mask_ins_targets:\n{}".format(mask_ins_targets))
                # print("mask_ins_masks:\n{}".format(mask_ins_masks))
                # print('inserted_tgt_tokens', inserted_tgt_tokens)

                # tok pred on exp_pld(input)
                word_ins_out, _ = self.decoder.forward_word_ins( # pred for all positions
                    normalize=False,
                    prev_output_tokens=masked_tgt_tokens,  # exp [pld]
                    encoder_out=encoder_out,
                )
                # print('after tok pred: ', torch.cuda.memory_allocated())

            if sampled_step == 1:
                output["sample1_mask_ins"] = {
                        "out": mask_ins_out,
                        "tgt": mask_ins_targets,
                        "mask": mask_ins_masks,
                        "ls": 0.01,
                    }
                
                output["sample1_word_ins"] = {
                    "out": word_ins_out,
                    "tgt": inserted_tgt_tokens,
                    "mask": masked_tgt_masks,
                    "ls": self.args.label_smoothing,
                    "nll_loss": True,
                }

            if new_del_input:
                # apply tok pred to get [empty -> exp pld -> tok pred] as input of deletion
                _, word_ins_pred = word_ins_out.max(-1)
                word_predictions, _ = _apply_ins_words(
                    masked_tgt_tokens.clone(),
                    None,
                    word_ins_pred,
                    None,
                    self.unk,
                )
                
            # deletion pred
            word_del_out, _ = self.decoder.forward_word_del(
                normalize=False,
                prev_output_tokens=word_predictions,
                encoder_out=encoder_out,
            )
            word_del_masks = word_predictions.ne(self.pad)

            # deletion label for training
            word_del_targets = _get_del_targets(word_predictions, tgt_tokens, self.pad)
            # print("word_del_targets:\n{}".format(word_del_targets))
            # #print("word_del_out:\n{}".format(word_del_out))
            # print("word_del_masks:\n{}".format(word_del_masks))
            # print('after del pred: ', torch.cuda.memory_allocated())

            # weight = torch.ones(word_del_out.size(-1), device=word_del_out.device).half()
            # weight[0] *= 0.5
            output["aggr_word_del"] = {
                    "out": word_del_out,
                    "tgt": word_del_targets,
                    "mask": word_del_masks,
                    "aggr_del": True,
                    # "weight": weight,
                }
            
            if sampled_step == 3:  # second decoding iteration
                # change delete threshold
                # delete_threshold = 0.2
                # if delete_threshold != 0.0:
                #     word_del_score = F.softmax(word_del_out, -1)
                #     word_del_pred = ((word_del_score[:,:, 1] - word_del_score[:,:, 0]) > delete_threshold).squeeze(-1)
                #     # print('avg score: ', torch.mean(word_del_score, dim=(0,1)).tolist())
                #     # print('after threshold: ', word_del_pred.sum().item())
                #     # print('original: ', word_del_score.max(-1)[1].bool().sum().item())
                # else:
                #     word_del_pred = word_del_score.max(-1)[1].bool()

                # apply deletion pred
                prev_output_tokens_applypred, _, _ = _apply_del_words(
                    word_predictions.clone(),
                    #word_predictions.clone(),
                    in_scores=None,
                    in_attn=None,
                    word_del_pred=word_del_out.max(-1)[1].bool(),
                    # word_del_pred=word_del_pred,
                    padding_idx=self.pad,
                    bos_idx=self.bos,
                    eos_idx=self.eos,
                )

                # delete unnecessary paddings
                cut_off = prev_output_tokens_applypred.ne(self.pad).sum(1).max()
                prev_output_tokens_applypred = prev_output_tokens_applypred[:, :cut_off]
                
                # torch.set_printoptions(profile="full")
                # print("prev_output_tokens_applypred:\n{}".format(prev_output_tokens_applypred))
                # print("tgt_tokens:\n{}".format(tgt_tokens))

                # generate training labels for insertion
                masked_tgt_masks, masked_tgt_tokens, mask_ins_targets, inserted_tgt_tokens = _get_ins_targets(
                    prev_output_tokens_applypred, tgt_tokens, self.pad, self.unk, aggravate=True
                )

                # cutoff unnecessary paddings
                cut_off = masked_tgt_tokens.ne(self.pad).sum(1).max()
                masked_tgt_tokens = masked_tgt_tokens[:, :cut_off]
                masked_tgt_masks = masked_tgt_masks[:, :cut_off]
                inserted_tgt_tokens = inserted_tgt_tokens[:, :cut_off]

                mask_ins_targets = mask_ins_targets.clamp(min=0, max=255)  # for safe prediction
                mask_ins_masks = prev_output_tokens_applypred[:, 1:].ne(self.pad)

                # print('cutoff: ', cut_off)
                # print('tgt len: ', tgt_tokens.ne(self.pad).sum(1).max())
                
                # print("masked_tgt_masks:\n{}".format(masked_tgt_masks))
                # #print("masked_tgt_tokens:\n{}".format(self.tgt_dict.string(masked_tgt_tokens)))
                # print("masked_tgt_tokens:\n{}".format(masked_tgt_tokens))
                # print("mask_ins_targets:\n{}".format(mask_ins_targets))
                # print("mask_ins_masks:\n{}".format(mask_ins_masks))

                # pld pred
                mask_ins_out, _ = self.decoder.forward_mask_ins(
                    normalize=False,
                    prev_output_tokens=prev_output_tokens_applypred,
                    encoder_out=encoder_out,
                )
                # print("mask_ins_out:\n{}".format(mask_ins_out))
                # print('after pld pred', torch.cuda.memory_allocated())

                # token pred
                word_ins_out, _ = self.decoder.forward_word_ins( # pred for all positions
                    normalize=False,
                    prev_output_tokens=masked_tgt_tokens,
                    encoder_out=encoder_out,
                )
                # print("word_ins_out:\n{}".format(word_ins_out))
                # print('after tok pred', torch.cuda.memory_allocated())

                output["sample3_mask_ins"] = {
                        "out": mask_ins_out,
                        "tgt": mask_ins_targets,
                        "mask": mask_ins_masks,
                        "ls": 0.01,
                        "sample3_mask": True,  # used when --save-tensors
                    }
                
                output["sample3_word_ins"] = {
                    "out": word_ins_out,
                    "tgt": inserted_tgt_tokens,
                    "mask": masked_tgt_masks,
                    "ls": self.args.label_smoothing,
                    "nll_loss": True,
                }

                # # apply tok pred to get [empty -> exp pld -> tok pred] as input of deletion
                # _, word_ins_pred = word_ins_out.max(-1)
                # word_predictions, _ = _apply_ins_words(
                #     masked_tgt_tokens.clone(),
                #     None,
                #     word_ins_pred,
                #     None,
                #     self.unk,
                # )
                
                # # deletion pred
                # word_del_out, _ = self.decoder.forward_word_del(
                #     normalize=False,
                #     prev_output_tokens=word_predictions,
                #     encoder_out=encoder_out,
                # )
                # word_del_masks = word_predictions.ne(self.pad)

                # # deletion label for training
                # word_del_targets = _get_del_targets(word_predictions, tgt_tokens, self.pad)
                # # print("word_del_targets:\n{}".format(word_del_targets))
                # # #print("word_del_out:\n{}".format(word_del_out))
                # # print("word_del_masks:\n{}".format(word_del_masks))
                # # print('after del pred: ', torch.cuda.memory_allocated())
            
                # output["sample3_word_del"] = {
                #         "out": word_del_out,
                #         "tgt": word_del_targets,
                #         "mask": word_del_masks,
                #         "aggr_del": True
                #     }
            

            return output

    def pre_predict(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, sampled_step, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # mask pred
        mask_ins_out, _ = self.decoder.forward_mask_ins(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )
        
        mask_ins_pred = mask_ins_out.max(-1)[1]
        masked_tgt_tokens_applypred, _ = _apply_ins_masks(  # change to pred [pld]
            prev_output_tokens.clone(),
            None,
            mask_ins_pred,
            self.pad,
            self.unk,
            self.eos,
        )       

        # token pred
        word_ins_out_applypred, _ = self.decoder.forward_word_ins( # pred for all positions
            normalize=False,
            prev_output_tokens=masked_tgt_tokens_applypred,  # pred [pld]
            encoder_out=encoder_out,
        )
        
        # apply token pred
        _, word_ins_pred = word_ins_out_applypred.max(-1)
        word_predictions, _ = _apply_ins_words(
            masked_tgt_tokens_applypred,
            None,
            word_ins_pred,
            None,
            self.unk,
        )

        # if sampled_step == 3:
        #     # deletion pred
        #     word_del_out, _ = self.decoder.forward_word_del(
        #         normalize=False,
        #         prev_output_tokens=word_predictions,
        #         encoder_out=encoder_out,
        #     )
        #     word_del_masks = word_predictions.ne(self.pad)

        #     # apply deletion pred
        #     prev_output_tokens_applypred, _, _ = _apply_del_words(
        #         word_predictions,
        #         #word_predictions.clone(),
        #         in_scores=None,
        #         in_attn=None,
        #         word_del_pred=word_del_out.max(-1)[1].bool(),
        #         # word_del_pred=word_del_pred,
        #         padding_idx=self.pad,
        #         bos_idx=self.bos,
        #         eos_idx=self.eos,
        #     )

        #     # delete unnecessary paddings
        #     cut_off = prev_output_tokens_applypred.ne(self.pad).sum(1).max()
        #     word_predictions = prev_output_tokens_applypred[:, :cut_off]
            
        
        return word_predictions
        

    def forward_decoder(
        self, decoder_out, encoder_out, eos_penalty=0.0, max_ratio=None, 
        delete_threshold=0.0, initial_ins_pred=None, step=-1, use_pld_dp=False, 
        initial_lens=None, initial_tokens=None, **kwargs
    ):
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        history = decoder_out.history

        bsz = output_tokens.size(0)
        if max_ratio is None:
            max_lens = torch.zeros_like(output_tokens).fill_(255)
        else:
            if not encoder_out["encoder_padding_mask"]:
                max_src_len = encoder_out["encoder_out"].size(0)
                src_lens = encoder_out["encoder_out"].new(bsz).fill_(max_src_len)
            else:
                src_lens = (~encoder_out["encoder_padding_mask"][0]).sum(1)
            max_lens = (src_lens * max_ratio).clamp(min=10).long()

        # delete words
        # do not delete tokens if it is <s> </s>
        can_del_word = output_tokens.ne(self.pad).sum(1) > 2
        if can_del_word.sum() != 0:  # we cannot delete, skip
            word_del_score, word_del_attn = self.decoder.forward_word_del(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_del_word),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_del_word),
            )

            # change delete threshold
            if delete_threshold != 0.0:
                word_del_score = F.softmax(word_del_score, -1)
                word_del_pred = ((word_del_score[:,:, 1] - word_del_score[:,:, 0]) > delete_threshold).squeeze(-1)
                # print('avg score: ', torch.mean(word_del_score, dim=(0,1)).tolist())
                # print('after threshold: ', word_del_pred.sum().item())
                # print('original: ', word_del_score.max(-1)[1].bool().sum().item())
            else:
                word_del_pred = word_del_score.max(-1)[1].bool()

            _tokens, _scores, _attn = _apply_del_words(
                output_tokens[can_del_word],
                output_scores[can_del_word],
                word_del_attn,
                word_del_pred,
                self.pad,
                self.bos,
                self.eos,
            )
            output_tokens = _fill(output_tokens, can_del_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_del_word, _scores, 0)
            attn = _fill(attn, can_del_word, _attn, 0.0)

            if history is not None:
                history.append(output_tokens.clone())
            
        # insert placeholders
        can_ins_mask = output_tokens.ne(self.pad).sum(1) < max_lens
        if can_ins_mask.sum() != 0:
            mask_ins_score, _ = self.decoder.forward_mask_ins(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_ins_mask),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_ins_mask),
            )
            if eos_penalty > 0.0:
                mask_ins_score[:, :, 0] = mask_ins_score[:, :, 0] - eos_penalty

            if use_pld_dp:
                if step == 0:
                    if initial_lens is not None:
                        mask_ins_pred = initial_lens
                        initial_ins_pred = initial_lens
                    else:
                        mask_ins_pred = mask_ins_score.max(-1)[1]
                        mask_ins_pred = torch.min(
                            mask_ins_pred, max_lens[can_ins_mask, None].expand_as(mask_ins_pred)
                        )
                        initial_ins_pred = mask_ins_pred
                else:
                    # print("output_tokens", output_tokens.shape, output_tokens)
                    # print("can_ins_mask", can_ins_mask.shape, can_ins_mask)
                    # print("initial_ins_pred", initial_ins_pred.shape, initial_ins_pred)
                    
                    mask_ins_pred = _get_pld_len(
                        mask_ins_score, 
                        output_tokens[can_ins_mask], 
                        initial_ins_pred[can_ins_mask], 
                        mask_ins_score.size(-1), 
                        self.pad)
            else:
                mask_ins_pred = mask_ins_score.max(-1)[1]
                mask_ins_pred = torch.min(
                    mask_ins_pred, max_lens[can_ins_mask, None].expand_as(mask_ins_pred)
                )
           
            _tokens, _scores = _apply_ins_masks(
                output_tokens[can_ins_mask],
                output_scores[can_ins_mask],
                mask_ins_pred,
                self.pad,
                self.unk,
                self.eos,
            )
            output_tokens = _fill(output_tokens, can_ins_mask, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_mask, _scores, 0)

            if history is not None:
                history.append(output_tokens.clone())

        # insert words
        can_ins_word = output_tokens.eq(self.unk).sum(1) > 0
        if can_ins_word.sum() != 0:
            word_ins_score, word_ins_attn = self.decoder.forward_word_ins(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_ins_word),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_ins_word),
            )
            word_ins_score, word_ins_pred = word_ins_score.max(-1)
            _tokens, _scores = _apply_ins_words(
                output_tokens[can_ins_word],
                output_scores[can_ins_word],
                word_ins_pred,
                word_ins_score,
                self.unk,
            )

            output_tokens = _fill(output_tokens, can_ins_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_word, _scores, 0)
            attn = _fill(attn, can_ins_word, word_ins_attn, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        attn = None if attn is None else attn[:, :cut_off, :]

        if use_pld_dp and step == 0:
            return decoder_out._replace(
                output_tokens=output_tokens,
                output_scores=output_scores,
                attn=attn,
                history=history,
            ), initial_ins_pred
        else:
            return decoder_out._replace(
                output_tokens=output_tokens,
                output_scores=output_scores,
                attn=attn,
                history=history,
            )


    def initialize_output_tokens(self, encoder_out, src_tokens, initial_tokens=None):
        ## xjt version: initial_tokens=prefix_tokens
        # initial_tokens = initial_tokens.tolist() if initial_tokens is not None else None
        # max_num_init_tok = max([len(seq) for seq in initial_tokens]) if initial_tokens is not None else 0
        # initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0), max_num_init_tok + 2)
        # initial_output_tokens[:, 0] = self.bos
        # initial_output_tokens[:, 1] = self.eos
        
        # if initial_tokens is not None:
        #     for i, seq in enumerate(initial_tokens):
        #         for j, tok in enumerate(seq):
        #             initial_output_tokens[i, j + 1] = tok
        #         initial_output_tokens[i, len(seq) + 1] = self.eos
        #         for j in range(len(seq) + 2, max_num_init_tok + 2):
        #             initial_output_tokens[i, j] = self.pad

        if initial_tokens is None:
            # empty at first: bos eos
            initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0), 2)  # B x 2
            initial_output_tokens[:, 0] = self.bos
            initial_output_tokens[:, 1] = self.eos
        else:
            initial_output_tokens = initial_tokens

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )


class LevenshteinTransformerDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
        self.embed_mask_ins = Embedding(256, self.output_embed_dim * 2, None)
        self.embed_word_del = Embedding(2, self.output_embed_dim, None)

        # del_word, ins_mask, ins_word
        self.early_exit = [int(i) for i in args.early_exit.split(",")]
        assert len(self.early_exit) == 3

        # copy layers for mask-predict/deletion
        self.layers_msk = None
        if getattr(args, "no_share_maskpredictor", False):
            self.layers_msk = nn.ModuleList(
                [
                    TransformerDecoderLayer(args, no_encoder_attn)
                    for _ in range(self.early_exit[1])
                ]
            )
        self.layers_del = None
        if getattr(args, "no_share_discriminator", False):
            self.layers_del = nn.ModuleList(
                [
                    TransformerDecoderLayer(args, no_encoder_attn)
                    for _ in range(self.early_exit[0])
                ]
            )

        if getattr(args, "share_discriminator_maskpredictor", False):
            assert getattr(
                args, "no_share_discriminator", False
            ), "must set saperate discriminator"
            self.layers_msk = self.layers_del

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        layers=None,
        **unused
    ):
        """
        Similar to *forward* but only return features.
        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        layers = self.layers if layers is None else layers
        early_exit = len(layers) if early_exit is None else early_exit
        for _, layer in enumerate(layers[:early_exit]):
            x, attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    @ensemble_decoder
    def forward_mask_ins(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[1],
            layers=self.layers_msk,
            **unused
        )
        features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        decoder_out = F.linear(features_cat, self.embed_mask_ins.weight)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra["attn"]

        return decoder_out, extra["attn"]

    @ensemble_decoder
    def forward_word_ins(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[2],
            layers=self.layers,
            **unused
        )
        decoder_out = self.output_layer(features)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra["attn"]
        return decoder_out, extra["attn"]

    @ensemble_decoder
    def forward_word_del(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[0],
            layers=self.layers_del,
            **unused
        )
        decoder_out = F.linear(features, self.embed_word_del.weight)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra["attn"]
        return decoder_out, extra["attn"]


@register_model_architecture("levenshtein_transformer", "levenshtein_transformer")
def levenshtein_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.early_exit = getattr(args, "early_exit", "6,6,6")
    args.no_share_discriminator = getattr(args, "no_share_discriminator", False)
    args.no_share_maskpredictor = getattr(args, "no_share_maskpredictor", False)
    args.share_discriminator_maskpredictor = getattr(
        args, "share_discriminator_maskpredictor", False
    )
    args.no_share_last_layer = getattr(args, "no_share_last_layer", False)
    args.prev_del = getattr(args, "prev_del", False)
    args.post_del = getattr(args, "post_del", False)

@register_model_architecture(
    "levenshtein_transformer", "levenshtein_transformer_wmt_en_de"
)
def levenshtein_transformer_wmt_en_de(args):
    levenshtein_base_architecture(args)


# similar parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture(
    "levenshtein_transformer", "levenshtein_transformer_vaswani_wmt_en_de_big"
)
def levenshtein_transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    levenshtein_base_architecture(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture(
    "levenshtein_transformer", "levenshtein_transformer_wmt_en_de_big"
)
def levenshtein_transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    levenshtein_transformer_vaswani_wmt_en_de_big(args)
