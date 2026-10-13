import json
import torch
import torch.nn.functional as F


def convert_sam_audio_state_dict(ckpt: dict, num_layers: int = 12) -> dict:
    """Converts upstream SAM-Audio PyTorch checkpoint to ComfyUI standard state dict.
    Collapses weight_norm (weight_g, weight_v) into standard weight tensors.
    """
    sd = {}

    def get_w(p: str):
        wg = ckpt.get(f"{p}.weight_g")
        wv = ckpt.get(f"{p}.weight_v")
        if wg is not None and wv is not None:
            return wg * F.normalize(wv, dim=[1, 2])
        return ckpt[f"{p}.weight"]

    def get_b(p: str):
        return ckpt.get(f"{p}.bias")

    # 1. audio_codec
    sd["audio_codec.encoder.in_conv.weight"] = get_w("audio_codec.encoder.block.0")
    sd["audio_codec.encoder.in_conv.bias"] = get_b("audio_codec.encoder.block.0")

    for bi in range(4):
        src_b = bi + 1
        tgt_b = f"audio_codec.encoder.blocks.{bi}"
        for ri, src_r in enumerate([0, 1, 2]):
            tgt_r = f"{tgt_b}.res{ri+1}"
            src_ru = f"audio_codec.encoder.block.{src_b}.block.{src_r}"
            sd[f"{tgt_r}.act1.alpha"] = ckpt[f"{src_ru}.block.0.alpha"]
            sd[f"{tgt_r}.conv1.weight"] = get_w(f"{src_ru}.block.1")
            sd[f"{tgt_r}.conv1.bias"] = get_b(f"{src_ru}.block.1")
            sd[f"{tgt_r}.act2.alpha"] = ckpt[f"{src_ru}.block.2.alpha"]
            sd[f"{tgt_r}.conv2.weight"] = get_w(f"{src_ru}.block.3")
            sd[f"{tgt_r}.conv2.bias"] = get_b(f"{src_ru}.block.3")
        sd[f"{tgt_b}.act.alpha"] = ckpt[f"audio_codec.encoder.block.{src_b}.block.3.alpha"]
        sd[f"{tgt_b}.conv.weight"] = get_w(f"audio_codec.encoder.block.{src_b}.block.4")
        sd[f"{tgt_b}.conv.bias"] = get_b(f"audio_codec.encoder.block.{src_b}.block.4")

    sd["audio_codec.encoder.out_act.alpha"] = ckpt["audio_codec.encoder.block.5.alpha"]
    sd["audio_codec.encoder.out_conv.weight"] = get_w("audio_codec.encoder.block.6")
    sd["audio_codec.encoder.out_conv.bias"] = get_b("audio_codec.encoder.block.6")

    sd["audio_codec.quantizer.in_proj.weight"] = get_w("audio_codec.quantizer.in_proj")
    sd["audio_codec.quantizer.in_proj.bias"] = get_b("audio_codec.quantizer.in_proj")
    sd["audio_codec.quantizer.out_proj.weight"] = get_w("audio_codec.quantizer.out_proj")
    sd["audio_codec.quantizer.out_proj.bias"] = get_b("audio_codec.quantizer.out_proj")

    sd["audio_codec.decoder.in_conv.weight"] = get_w("audio_codec.decoder.model.0")
    sd["audio_codec.decoder.in_conv.bias"] = get_b("audio_codec.decoder.model.0")

    res_indices = [(4, 1), (5, 2), (8, 3)]
    for bi in range(4):
        src_b = bi + 1
        tgt_b = f"audio_codec.decoder.blocks.{bi}"
        sd[f"{tgt_b}.act.alpha"] = ckpt[f"audio_codec.decoder.model.{src_b}.block.0.alpha"]
        sd[f"{tgt_b}.conv_tr.weight"] = get_w(f"audio_codec.decoder.model.{src_b}.block.1")
        sd[f"{tgt_b}.conv_tr.bias"] = get_b(f"audio_codec.decoder.model.{src_b}.block.1")
        for src_ru_idx, tgt_ru_idx in res_indices:
            tgt_r = f"{tgt_b}.res{tgt_ru_idx}"
            src_ru = f"audio_codec.decoder.model.{src_b}.block.{src_ru_idx}"
            sd[f"{tgt_r}.act1.alpha"] = ckpt[f"{src_ru}.block.0.alpha"]
            sd[f"{tgt_r}.conv1.weight"] = get_w(f"{src_ru}.block.1")
            sd[f"{tgt_r}.conv1.bias"] = get_b(f"{src_ru}.block.1")
            sd[f"{tgt_r}.act2.alpha"] = ckpt[f"{src_ru}.block.2.alpha"]
            sd[f"{tgt_r}.conv2.weight"] = get_w(f"{src_ru}.block.3")
            sd[f"{tgt_r}.conv2.bias"] = get_b(f"{src_ru}.block.3")

    sd["audio_codec.decoder.out_act.alpha"] = ckpt["audio_codec.decoder.wm_model.encoder_block.pre.0.alpha"]
    sd["audio_codec.decoder.out_conv.weight"] = get_w("audio_codec.decoder.wm_model.encoder_block.pre.1")
    sd["audio_codec.decoder.out_conv.bias"] = get_b("audio_codec.decoder.wm_model.encoder_block.pre.1")

    # 2. transformer
    sd["transformer.x_embedder.block1_gn.weight"] = ckpt["transformer.x_embedder.block.block1.groupnorm.weight"]
    sd["transformer.x_embedder.block1_gn.bias"] = ckpt["transformer.x_embedder.block.block1.groupnorm.bias"]
    sd["transformer.x_embedder.block1_conv.weight"] = ckpt["transformer.x_embedder.block.block1.project.weight"]
    sd["transformer.x_embedder.block1_conv.bias"] = ckpt["transformer.x_embedder.block.block1.project.bias"]
    sd["transformer.x_embedder.block2_gn.weight"] = ckpt["transformer.x_embedder.block.block2.groupnorm.weight"]
    sd["transformer.x_embedder.block2_gn.bias"] = ckpt["transformer.x_embedder.block.block2.groupnorm.bias"]
    sd["transformer.x_embedder.block2_conv.weight"] = ckpt["transformer.x_embedder.block.block2.project.weight"]
    sd["transformer.x_embedder.block2_conv.bias"] = ckpt["transformer.x_embedder.block.block2.project.bias"]

    sd["transformer.t_embedder.w1.weight"] = ckpt["transformer.t_embedder.projection.w1.weight"]
    sd["transformer.t_embedder.w2.weight"] = ckpt["transformer.t_embedder.projection.w2.weight"]
    sd["transformer.t_embedder.w3.weight"] = ckpt["transformer.t_embedder.projection.w3.weight"]

    sd["transformer.t_block.weight"] = ckpt["transformer.t_block.weight"]
    sd["transformer.t_block.bias"] = ckpt["transformer.t_block.bias"]

    sd["transformer.y_embedder.w1.weight"] = ckpt["transformer.y_embedder.projection.w1.weight"]
    sd["transformer.y_embedder.w2.weight"] = ckpt["transformer.y_embedder.projection.w2.weight"]
    sd["transformer.y_embedder.w3.weight"] = ckpt["transformer.y_embedder.projection.w3.weight"]

    sd["transformer.norm.weight"] = ckpt["transformer.norm.weight"]
    sd["transformer.output.weight"] = ckpt["transformer.output.weight"]
    sd["transformer.final_layer_scale_shift_table"] = ckpt["transformer.final_layer_scale_shift_table"]

    for i in range(num_layers):
        src = f"transformer.layers.{i}"
        tgt = f"transformer.layers.{i}"
        sd[f"{tgt}.scale_shift_table"] = ckpt[f"{src}.scale_shift_table"]
        sd[f"{tgt}.attention_norm.weight"] = ckpt[f"{src}.attention_norm.weight"]
        sd[f"{tgt}.ffn_norm.weight"] = ckpt[f"{src}.ffn_norm.weight"]

        sd[f"{tgt}.attention.wq.weight"] = ckpt[f"{src}.attention.wq.weight"]
        sd[f"{tgt}.attention.wk.weight"] = ckpt[f"{src}.attention.wk.weight"]
        sd[f"{tgt}.attention.wv.weight"] = ckpt[f"{src}.attention.wv.weight"]
        sd[f"{tgt}.attention.wo.weight"] = ckpt[f"{src}.attention.wo.weight"]
        sd[f"{tgt}.attention.q_norm.weight"] = ckpt[f"{src}.attention.q_norm.weight"]
        sd[f"{tgt}.attention.k_norm.weight"] = ckpt[f"{src}.attention.k_norm.weight"]

        sd[f"{tgt}.cross_attention.wq.weight"] = ckpt[f"{src}.cross_attention.wq.weight"]
        sd[f"{tgt}.cross_attention.wk.weight"] = ckpt[f"{src}.cross_attention.wk.weight"]
        sd[f"{tgt}.cross_attention.wv.weight"] = ckpt[f"{src}.cross_attention.wv.weight"]
        sd[f"{tgt}.cross_attention.wo.weight"] = ckpt[f"{src}.cross_attention.wo.weight"]
        sd[f"{tgt}.cross_attention.q_norm.weight"] = ckpt[f"{src}.cross_attention.q_norm.weight"]
        sd[f"{tgt}.cross_attention.k_norm.weight"] = ckpt[f"{src}.cross_attention.k_norm.weight"]

        sd[f"{tgt}.feed_forward.w1.weight"] = ckpt[f"{src}.feed_forward.w1.weight"]
        sd[f"{tgt}.feed_forward.w2.weight"] = ckpt[f"{src}.feed_forward.w2.weight"]
        sd[f"{tgt}.feed_forward.w3.weight"] = ckpt[f"{src}.feed_forward.w3.weight"]

    # 3. projections and alignment
    sd["proj.weight"] = ckpt["proj.weight"]
    sd["proj.bias"] = ckpt["proj.bias"]
    sd["memory_proj.weight"] = ckpt["memory_proj.weight"]
    sd["memory_proj.bias"] = ckpt["memory_proj.bias"]

    sd["align_masked_video.conv.weight"] = ckpt["align_masked_video.conv.weight"]
    sd["align_masked_video.conv.bias"] = ckpt["align_masked_video.conv.bias"]
    sd["align_masked_video.layer_norm.weight"] = ckpt["align_masked_video.layer_norm.weight"]
    sd["align_masked_video.layer_norm.bias"] = ckpt["align_masked_video.layer_norm.bias"]
    sd["align_masked_video.gate"] = ckpt["align_masked_video.gate"]

    sd["embed_anchors.embed.weight"] = ckpt["embed_anchors.embed.weight"]
    sd["embed_anchors.proj.weight"] = ckpt["embed_anchors.proj.weight"]
    sd["embed_anchors.gate"] = ckpt["embed_anchors.gate"]

    # 4. vision encoder
    v_prefix = "vision_encoder.model.visual."
    t_v = "vision_encoder.visual."
    sd[f"{t_v}conv1.weight"] = ckpt[f"{v_prefix}conv1.weight"]
    sd[f"{t_v}class_embedding"] = ckpt[f"{v_prefix}class_embedding"]
    sd[f"{t_v}positional_embedding"] = ckpt[f"{v_prefix}positional_embedding"]
    sd[f"{t_v}ln_pre.weight"] = ckpt[f"{v_prefix}ln_pre.weight"]
    sd[f"{t_v}ln_pre.bias"] = ckpt[f"{v_prefix}ln_pre.bias"]

    for i in range(24):
        src_rb = f"{v_prefix}transformer.resblocks.{i}."
        tgt_rb = f"{t_v}resblocks.{i}."
        sd[f"{tgt_rb}ln_1.weight"] = ckpt[f"{src_rb}ln_1.weight"]
        sd[f"{tgt_rb}ln_1.bias"] = ckpt[f"{src_rb}ln_1.bias"]
        sd[f"{tgt_rb}in_proj.weight"] = ckpt[f"{src_rb}attn.in_proj_weight"]
        sd[f"{tgt_rb}in_proj.bias"] = ckpt[f"{src_rb}attn.in_proj_bias"]
        sd[f"{tgt_rb}out_proj.weight"] = ckpt[f"{src_rb}attn.out_proj.weight"]
        sd[f"{tgt_rb}out_proj.bias"] = ckpt[f"{src_rb}attn.out_proj.bias"]
        sd[f"{tgt_rb}ln_2.weight"] = ckpt[f"{src_rb}ln_2.weight"]
        sd[f"{tgt_rb}ln_2.bias"] = ckpt[f"{src_rb}ln_2.bias"]
        sd[f"{tgt_rb}c_fc.weight"] = ckpt[f"{src_rb}mlp.c_fc.weight"]
        sd[f"{tgt_rb}c_fc.bias"] = ckpt[f"{src_rb}mlp.c_fc.bias"]
        sd[f"{tgt_rb}c_proj.weight"] = ckpt[f"{src_rb}mlp.c_proj.weight"]
        sd[f"{tgt_rb}c_proj.bias"] = ckpt[f"{src_rb}mlp.c_proj.bias"]

    sd[f"{t_v}ln_post.weight"] = ckpt[f"{v_prefix}ln_post.weight"]
    sd[f"{t_v}ln_post.bias"] = ckpt[f"{v_prefix}ln_post.bias"]
    sd[f"{t_v}attn_pool.probe"] = ckpt[f"{v_prefix}attn_pool.probe"]
    sd[f"{t_v}attn_pool.in_proj.weight"] = ckpt[f"{v_prefix}attn_pool.attn.in_proj_weight"]
    sd[f"{t_v}attn_pool.in_proj.bias"] = ckpt[f"{v_prefix}attn_pool.attn.in_proj_bias"]
    sd[f"{t_v}attn_pool.out_proj.weight"] = ckpt[f"{v_prefix}attn_pool.attn.out_proj.weight"]
    sd[f"{t_v}attn_pool.out_proj.bias"] = ckpt[f"{v_prefix}attn_pool.attn.out_proj.bias"]
    sd[f"{t_v}attn_pool.layernorm.weight"] = ckpt[f"{v_prefix}attn_pool.layernorm.weight"]
    sd[f"{t_v}attn_pool.layernorm.bias"] = ckpt[f"{v_prefix}attn_pool.layernorm.bias"]
    sd[f"{t_v}attn_pool.c_fc.weight"] = ckpt[f"{v_prefix}attn_pool.mlp.c_fc.weight"]
    sd[f"{t_v}attn_pool.c_fc.bias"] = ckpt[f"{v_prefix}attn_pool.mlp.c_fc.bias"]
    sd[f"{t_v}attn_pool.c_proj.weight"] = ckpt[f"{v_prefix}attn_pool.mlp.c_proj.weight"]
    sd[f"{t_v}attn_pool.c_proj.bias"] = ckpt[f"{v_prefix}attn_pool.mlp.c_proj.bias"]
    sd[f"{t_v}proj"] = ckpt[f"{v_prefix}proj"]

    return sd


def is_raw_sam_audio_checkpoint(sd: dict) -> bool:
    return "audio_codec.encoder.block.0.weight_g" in sd or "transformer.x_embedder.block.block1.groupnorm.weight" in sd
