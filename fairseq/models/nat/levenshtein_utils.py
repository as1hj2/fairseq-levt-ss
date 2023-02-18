# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq.utils import new_arange


# -------------- Helper Functions --------------------------------------------------- #


def load_libnat():
    try:
        from fairseq import libnat_cuda

        return libnat_cuda, True

    except ImportError as e:
        print(str(e) + "... fall back to CPU version")

        try:
            from fairseq import libnat

            return libnat, False

        except ImportError as e:
            import sys

            sys.stderr.write(
                "ERROR: missing libnat_cuda. run `python setup.py build_ext --inplace`\n"
            )
            raise e

def _get_pld_len(mask_ins_score, output_tokens, initial_ins_pred, K, pad):
    from fairseq import libnat
    #print("output tokens: ", output_tokens)
    output_masks = output_tokens.ne(pad)
    Ls = output_masks.sum(dim=-1)
    mask_ins_proba = torch.nn.functional.softmax(mask_ins_score, dim=-1)
    Ps = -torch.log(mask_ins_proba.float())

    # print("output tokens: ", output_tokens.shape, output_tokens)
    # print("Ls: ", Ls)
    # print("obj: ", initial_ins_pred)
    # print("Ps: ", Ps.shape)
    
    full_preds = libnat.suggested_pld_len(Ps.tolist(), Ls.tolist(), initial_ins_pred.squeeze(dim=-1).tolist(), K)  # (batch * Ls-1, batch * Ls-1)
    # print("full preds", full_preds)
    predecessors = torch.tensor(full_preds[0], dtype=output_tokens.dtype, device=output_tokens.device)
    costs =full_preds[1]

    # print("pld preds: ", predecessors.shape, predecessors)

    return predecessors

def _get_ins_targets(in_tokens, out_tokens, padding_idx, unk_idx, aggravate=False):
    libnat, use_cuda = load_libnat()

    def _get_ins_targets_cuda(in_tokens, out_tokens, padding_idx, unk_idx):
        in_masks = in_tokens.ne(padding_idx)
        out_masks = out_tokens.ne(padding_idx)

        # print("in_tokens: ", in_tokens)

        if aggravate:
            mask_ins_targets, masked_tgt_masks, tgt_masks = libnat.generate_insertion_labels_aggravate(
                out_tokens.int(),
                libnat.levenshtein_distance(
                    in_tokens.int(),
                    out_tokens.int(),
                    in_masks.sum(1).int(),
                    out_masks.sum(1).int(),
                ),
            )
            
            # print('cuda output mask_ins_targets, masked_tgt_masks, tgt_masks: ', mask_ins_targets, masked_tgt_masks, tgt_masks)
        
            masked_tgt_masks = masked_tgt_masks.bool()
            tgt_masks = tgt_masks.bool()
            mask_ins_targets = mask_ins_targets.type_as(in_tokens)[:, 1 : in_masks.size(1)]

            masked_tgt_tokens = torch.full(  # len = in + out
                masked_tgt_masks.size(), 
                padding_idx,  # initial: all pad
                dtype=out_tokens.dtype, 
                device=out_tokens.device
                )

            masked_tgt_tokens[masked_tgt_masks] = unk_idx  # insert pld
            
            # insert input tokens
            range_tensor = torch.arange(masked_tgt_masks.size(1), device=in_masks.device).unsqueeze(0)
            range_tensor = range_tensor.expand(masked_tgt_masks.size(0), range_tensor.size(1))
            remove_padding_mask = range_tensor < (in_masks.sum(1, keepdim=True) + mask_ins_targets.sum(1, keepdim=True))
            
            torch.set_printoptions(profile="full")
            # print("remove_padding_mask:\n{}".format(remove_padding_mask))

            masked_tgt_tokens.masked_scatter_(
                    masked_tgt_masks ^ remove_padding_mask,
                    in_tokens[in_masks]
                )
            
            # print('masked_tgt_masks', masked_tgt_masks)
            # print('masked_tgt_tokens', masked_tgt_tokens)

            inserted_tgt_tokens = masked_tgt_tokens.clone()
            inserted_tgt_tokens.masked_scatter_(masked_tgt_masks, out_tokens[tgt_masks])

        else:
            mask_ins_targets, masked_tgt_masks = libnat.generate_insertion_labels(
                out_tokens.int(),
                libnat.levenshtein_distance(
                    in_tokens.int(),
                    out_tokens.int(),
                    in_masks.sum(1).int(),
                    out_masks.sum(1).int(),
                ),
            )
            
            inserted_tgt_tokens = None
            masked_tgt_masks = masked_tgt_masks.bool() & out_masks
            mask_ins_targets = mask_ins_targets.type_as(in_tokens)[
                :, 1 : in_masks.size(1)
            ].masked_fill_(~in_masks[:, 1:], 0)
            masked_tgt_tokens = out_tokens.masked_fill(masked_tgt_masks, unk_idx)

        return masked_tgt_masks, masked_tgt_tokens, mask_ins_targets, inserted_tgt_tokens

    def _get_ins_targets_cpu(in_tokens, out_tokens, padding_idx, unk_idx):
        in_seq_len, out_seq_len = in_tokens.size(1), out_tokens.size(1)

        in_tokens_list = [
            [t for t in s if t != padding_idx] for i, s in enumerate(in_tokens.tolist())
        ]
        out_tokens_list = [
            [t for t in s if t != padding_idx]
            for i, s in enumerate(out_tokens.tolist())
        ]

        full_labels = libnat.suggested_ed2_path(
            in_tokens_list, out_tokens_list, padding_idx
        )
        mask_inputs = [
            [len(c) if c[0] != padding_idx else 0 for c in a[:-1]] for a in full_labels
        ]

        # generate labels
        inserted_tgt_tokens = None
        if not aggravate:
            masked_tgt_masks = []
            for mask_input in mask_inputs:
                mask_label = []
                for beam_size in mask_input[1:-1]:  # HACK 1:-1
                    mask_label += [0] + [1 for _ in range(beam_size)]
                masked_tgt_masks.append(
                    mask_label + [0 for _ in range(out_seq_len - len(mask_label))]
                )
            mask_ins_targets = [
                mask_input[1:-1]
                + [0 for _ in range(in_seq_len - 1 - len(mask_input[1:-1]))]
                for mask_input in mask_inputs
            ]
            # transform to tensor
            masked_tgt_masks = torch.tensor(
                masked_tgt_masks, device=out_tokens.device
            ).bool()
            mask_ins_targets = torch.tensor(mask_ins_targets, device=in_tokens.device)
            masked_tgt_tokens = out_tokens.masked_fill(masked_tgt_masks, unk_idx)

            #return masked_tgt_masks, masked_tgt_tokens, mask_ins_targets

        else: # if use aggravate, input is no longer a subsequence
            masked_tgt_masks = []
            masked_tgt_tokens = []
            inserted_tgt_tokens = []
            pad_to_len = 0
            for idx, mask_input in enumerate(mask_inputs):
                mask_label = []
                masked_tgt_token = []
                inserted_tgt_token = []
                idx_input = 0
                # note that the result is without eos and pad
                for b_idx, beam_size in enumerate(mask_input[1:-1]):  # HACK 1:-1, i.e. remove bos, eos
                    mask_label += [0] + [1 for _ in range(beam_size)]  # eg. ins 0220 -> 0 011 011 0
                    masked_tgt_token += [in_tokens_list[idx][idx_input]] + [unk_idx for _ in range(beam_size)]
                    inserted_tgt_token += [in_tokens_list[idx][idx_input]]
                    if beam_size != 0:
                        inserted_tgt_token += full_labels[idx][1 + b_idx]
                    idx_input += 1
                masked_tgt_masks.append(mask_label)
                masked_tgt_tokens.append(masked_tgt_token)
                inserted_tgt_tokens.append(inserted_tgt_token)
                if len(masked_tgt_token) > pad_to_len:
                    pad_to_len = len(masked_tgt_token)
            # pad to max len after insertion, maybe longer than input and output
            masked_tgt_masks = [mask_label + [2] + [0 for _ in range(pad_to_len - len(mask_label))] for mask_label in masked_tgt_masks]
            masked_tgt_tokens = [masked_tgt_token + [2] + [0 for _ in range(pad_to_len - len(masked_tgt_token))] for masked_tgt_token in masked_tgt_tokens]
            inserted_tgt_tokens = [inserted_tgt_token + [2] + [0 for _ in range(pad_to_len - len(inserted_tgt_token))] for inserted_tgt_token in inserted_tgt_tokens]

            mask_ins_targets = [ # len = len(input)-1, later let pred also same len, easy for loss
                mask_input[1:-1]
                + [0 for _ in range(in_seq_len - 1 - len(mask_input[1:-1]))]
                for mask_input in mask_inputs
            ]

            # transform to tensor
            masked_tgt_masks = torch.tensor(
                masked_tgt_masks, device=out_tokens.device
            ).bool()
            mask_ins_targets = torch.tensor(mask_ins_targets, device=in_tokens.device)
            masked_tgt_tokens = torch.tensor(masked_tgt_tokens, device=in_tokens.device)
            inserted_tgt_tokens = torch.tensor(inserted_tgt_tokens, device=in_tokens.device)

        return masked_tgt_masks, masked_tgt_tokens, mask_ins_targets, inserted_tgt_tokens

    if use_cuda:
        return _get_ins_targets_cuda(in_tokens, out_tokens, padding_idx, unk_idx)
    return _get_ins_targets_cpu(in_tokens, out_tokens, padding_idx, unk_idx)


def _get_del_targets(in_tokens, out_tokens, padding_idx):
    libnat, use_cuda = load_libnat()

    def _get_del_targets_cuda(in_tokens, out_tokens, padding_idx):
        in_masks = in_tokens.ne(padding_idx)
        out_masks = out_tokens.ne(padding_idx)

        word_del_targets = libnat.generate_deletion_labels(
            in_tokens.int(),
            libnat.levenshtein_distance(
                in_tokens.int(),
                out_tokens.int(),
                in_masks.sum(1).int(),
                out_masks.sum(1).int(),
            ),
        )
        word_del_targets = word_del_targets.type_as(in_tokens).masked_fill_(
            ~in_masks, 0
        )
        return word_del_targets

    def _get_del_targets_cpu(in_tokens, out_tokens, padding_idx):
        out_seq_len = out_tokens.size(1)
        with torch.cuda.device_of(in_tokens):
            in_tokens_list = [
                [t for t in s if t != padding_idx]
                for i, s in enumerate(in_tokens.tolist())
            ]
            out_tokens_list = [
                [t for t in s if t != padding_idx]
                for i, s in enumerate(out_tokens.tolist())
            ]

        full_labels = libnat.suggested_ed2_path(
            in_tokens_list, out_tokens_list, padding_idx
        )
        word_del_targets = [b[-1] for b in full_labels]
        word_del_targets = [
            labels + [0 for _ in range(out_seq_len - len(labels))]
            for labels in word_del_targets
        ]

        # transform to tensor
        word_del_targets = torch.tensor(word_del_targets, device=out_tokens.device)
        return word_del_targets

    if use_cuda:
        return _get_del_targets_cuda(in_tokens, out_tokens, padding_idx)
    return _get_del_targets_cpu(in_tokens, out_tokens, padding_idx)


def _apply_ins_masks(
    in_tokens, in_scores, mask_ins_pred, padding_idx, unk_idx, eos_idx
):

    in_masks = in_tokens.ne(padding_idx)
    in_lengths = in_masks.sum(1)

    # HACK: hacky way to shift all the paddings to eos first.
    in_tokens.masked_fill_(~in_masks, eos_idx)
    mask_ins_pred.masked_fill_(~in_masks[:, 1:], 0)

    out_lengths = in_lengths + mask_ins_pred.sum(1)
    out_max_len = out_lengths.max()
    out_masks = new_arange(out_lengths, out_max_len)[None, :] < out_lengths[:, None]

    reordering = (mask_ins_pred + in_masks[:, 1:].long()).cumsum(1)
    out_tokens = (
        in_tokens.new_zeros(in_tokens.size(0), out_max_len)
        .fill_(padding_idx)
        .masked_fill_(out_masks, unk_idx)
    )
    out_tokens[:, 0] = in_tokens[:, 0]
    out_tokens.scatter_(1, reordering, in_tokens[:, 1:])

    out_scores = None
    if in_scores is not None:
        in_scores.masked_fill_(~in_masks, 0)
        out_scores = in_scores.new_zeros(*out_tokens.size())
        out_scores[:, 0] = in_scores[:, 0]
        out_scores.scatter_(1, reordering, in_scores[:, 1:])

    return out_tokens, out_scores


def _apply_ins_words(in_tokens, in_scores, word_ins_pred, word_ins_scores, unk_idx):
    word_ins_masks = in_tokens.eq(unk_idx)
    out_tokens = in_tokens.masked_scatter(word_ins_masks, word_ins_pred[word_ins_masks])

    if in_scores is not None:
        out_scores = in_scores.masked_scatter(
            word_ins_masks, word_ins_scores[word_ins_masks]
        )
    else:
        out_scores = None

    return out_tokens, out_scores


def _apply_del_words(
    in_tokens, in_scores, in_attn, word_del_pred, padding_idx, bos_idx, eos_idx
):
    # apply deletion to a tensor
    in_masks = in_tokens.ne(padding_idx)
    bos_eos_masks = in_tokens.eq(bos_idx) | in_tokens.eq(eos_idx)

    max_len = in_tokens.size(1)
    word_del_pred.masked_fill_(~in_masks, 1)
    word_del_pred.masked_fill_(bos_eos_masks, 0)

    reordering = new_arange(in_tokens).masked_fill_(word_del_pred, max_len).sort(1)[1]

    out_tokens = in_tokens.masked_fill(word_del_pred, padding_idx).gather(1, reordering)

    out_scores = None
    if in_scores is not None:
        out_scores = in_scores.masked_fill(word_del_pred, 0).gather(1, reordering)

    out_attn = None
    if in_attn is not None:
        _mask = word_del_pred[:, :, None].expand_as(in_attn)
        _reordering = reordering[:, :, None].expand_as(in_attn)
        out_attn = in_attn.masked_fill(_mask, 0.0).gather(1, _reordering)

    return out_tokens, out_scores, out_attn


def _skip(x, mask):
    """
    Getting sliced (dim=0) tensor by mask. Supporting tensor and list/dict of tensors.
    """
    if isinstance(x, int):
        return x

    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        if x.size(0) == mask.size(0):
            return x[mask]
        elif x.size(1) == mask.size(0):
            return x[:, mask]

    if isinstance(x, list):
        return [_skip(x_i, mask) for x_i in x]

    if isinstance(x, dict):
        return {k: _skip(v, mask) for k, v in x.items()}

    raise NotImplementedError


def _skip_encoder_out(encoder, encoder_out, mask):
    if not mask.any():
        return encoder_out
    else:
        return encoder.reorder_encoder_out(
            encoder_out, mask.nonzero(as_tuple=False).squeeze()
        )


def _fill(x, mask, y, padding_idx):
    """
    Filling tensor x with y at masked positions (dim=0).
    """
    if x is None:
        return y
    assert x.dim() == y.dim() and mask.size(0) == x.size(0)
    assert x.dim() == 2 or (x.dim() == 3 and x.size(2) == y.size(2))
    n_selected = mask.sum()
    assert n_selected == y.size(0)

    if n_selected == x.size(0):
        return y

    if x.size(1) < y.size(1):
        dims = [x.size(0), y.size(1) - x.size(1)]
        if x.dim() == 3:
            dims.append(x.size(2))
        x = torch.cat([x, x.new_zeros(*dims).fill_(padding_idx)], 1)
        x[mask] = y
    elif x.size(1) > y.size(1):
        x[mask] = padding_idx
        if x.dim() == 2:
            x[mask, : y.size(1)] = y
        else:
            x[mask, : y.size(1), :] = y
    else:
        x[mask] = y
    return x
