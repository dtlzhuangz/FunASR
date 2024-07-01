#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import types
import torch
from funasr.register import tables


def export_rebuild_model(model, **kwargs):
    model.device = kwargs.get("device")
    is_onnx = kwargs.get("type", "onnx") == "onnx"
    encoder_class = tables.encoder_classes.get(kwargs["encoder"] + "Export")
    model.encoder = encoder_class(model.encoder, onnx=is_onnx)

    predictor_class = tables.predictor_classes.get(kwargs["predictor"] + "Export")
    model.predictor = predictor_class(model.predictor, onnx=is_onnx)

    decoder_class = tables.decoder_classes.get(kwargs["decoder"] + "Export")
    model.decoder = decoder_class(model.decoder, onnx=is_onnx)

    from funasr.utils.torch_function import sequence_mask

    model.make_pad_mask = sequence_mask(kwargs["max_seq_len"], flip=False)

    model.forward = types.MethodType(export_forward, model)
    model.export_dummy_inputs = types.MethodType(export_dummy_inputs, model)
    model.export_input_names = types.MethodType(export_input_names, model)
    model.export_output_names = types.MethodType(export_output_names, model)
    model.export_dynamic_axes = types.MethodType(export_dynamic_axes, model)
    model.export_name = types.MethodType(export_name, model)
    return model


def export_forward(
    self,
    enc: torch.Tensor,
    enc_len: torch.Tensor,
):
    # a. To device
    # batch = {"speech": speech, "speech_lengths": speech_lengths}
    # batch = to_device(batch, device=self.device)

    # enc, enc_len = self.encoder(**batch)
    mask = self.make_pad_mask(enc_len, max_seq_len=enc.size(1))[:, None, :]
    pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = self.predictor(
        enc, mask
    )
    pre_token_length = pre_token_length.floor().type(torch.int32)

    decoder_out, _ = self.decoder(enc, enc_len, pre_acoustic_embeds, pre_token_length)
    decoder_out = torch.log_softmax(decoder_out, dim=-1)
    # sample_ids = decoder_out.argmax(dim=-1)

    return decoder_out, pre_token_length


def export_dummy_inputs(self):
    import numpy as np

    enc = torch.from_numpy(np.load("/mnt/local-storage/zhuangzhong/FunASR/enc.npy"))
    enc_len = torch.from_numpy(
        np.load("/mnt/local-storage/zhuangzhong/FunASR/enc_len.npy")
    )
    # speech = torch.randn(1, 50, 560)
    # speech_lengths = torch.tensor([1], dtype=torch.int32)
    return (enc, enc_len)


def export_input_names(self):
    return ["enc", "enc_len"]


def export_output_names(self):
    return ["logits", "token_num"]


def export_dynamic_axes(self):
    return {
        "enc": {0: "batch_size", 1: "feats_length"},
        "enc_len": {0: "batch_size"},
        "logits": {0: "batch_size", 1: "logits_length"},
    }


def export_name(
    self,
):
    return "model.onnx"
